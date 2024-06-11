from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance 
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import google.generativeai as gen_ai
import os
import uuid
from dotenv import load_dotenv
import re
import uvicorn


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
gemini_model = gen_ai.GenerativeModel("gemini-pro")

# Initialize FastAPI app
app = FastAPI()

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-t5-base')
COLLECTION_NAME = "youtube_transcriptions"

# Ensure the collection exists
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)



class YouTubeURL(BaseModel):
    youtube_url: str

class Query(BaseModel):
    query: str    


@app.post("/transcription")
async def get_transcription(youtube_url: YouTubeURL):
    video_id = extract_video_id(youtube_url.youtube_url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        chunks = split_text_into_chunks(transcript_text, 2000)
        embeddings = generate_embeddings(chunks)
        points = create_points(chunks, embeddings, youtube_url.youtube_url)
        formatted_chunks = [[chunk] for chunk in chunks] 
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
        return {"chunks": formatted_chunks, "embeddings": embeddings}

    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

def extract_video_id(url: str) -> str:
    
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


def split_text_into_chunks(text: str, max_chunk_size: int):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings.tolist()  # Convert numpy array to list for JSON serialization
    

def create_points(chunks, embeddings, url):
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={ "url": url, "chunk": chunk}
        )
        points.append(point)
    return points




    
@app.post("/similarity_search")
async def similarity_search(query: Query):
    try:
         # Encode the query chunk
        query_embedding = model.encode([query.query])[0]
        
        # Define search parameters to retrieve top 2 similar chunks
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),  # Convert to list
            limit=2  # Retrieve top 2 similar chunks
        )

        results = [{"chunk": point.payload["chunk"], "score": point.score} for point in search_results]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

    
@app.post("/model")
async def similarity_search(query: Query):
    
        # Encode the query chunk
        query_embedding = model.encode([query.query])[0]
        
        # Define search parameters to retrieve top 2 similar chunks
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),  # Convert to list
            limit=2  # Retrieve top 2 similar chunks
        )

        results = [{"chunk": point.payload["chunk"], "score": point.score} for point in search_results]
        context = " ".join([point["chunk"] for point in results])
        
        prompt = f"""Answer the question as detailed as possible from the provided context, make sure to provide all the details 
                    and provide minimum 2 lines of context. If the answer is not available in the context, answer was not found in db.\n\n
                    Context:\n{context}\n
                    Question:\n{query.query}\n
                """
        try:
          response = gen_ai.generate_text(prompt=prompt)
          answer = response.result  # Extract the actual answer from the response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

        return {"Response": answer}     
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    
