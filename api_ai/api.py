import pickle
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
from typing import List, Dict, Optional
# from generative_resp.ai_response import send_response, create_index, update_vector_store_with_new_file
from generative_resp.ai_response import (
    send_response, 
    load_base_vector_store, 
    update_in_memory_vector_store
)

import shutil
import os
import uuid

app = FastAPI()

# --- SESSION MANAGEMENT ---

# 1. Load the vector_store the first time initializing the app
print("Loading base vector store at startup...")
BASE_VECTOR_STORE = load_base_vector_store()
print("Base vector store loaded.")

# 2. Dictionary in memory to store the vector stores for each session
# The key will be the session_id, and the value will be the FAISS object
SESSION_VECTOR_STORES: Dict[str, object] = {}

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models for request validation
class Message(BaseModel):
    role: str  # "human" or "ai"
    content: str

class QuestionRequest(BaseModel):
    question: str
    session_id: str 
    conversation_history: Optional[List[Message]] = []

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint for asking questions with optional conversation history.
    This endpoint now ensures that every session has its own isolated vector store.
    """
    try:
        session_id = request.session_id
        # First time session, create his own in-memory vector DB
        if session_id not in SESSION_VECTOR_STORES:
            print(f"Creating new in-memory vector store for session: {session_id}")
            # Make a copy of the base vector DB
            cloned_store = pickle.loads(pickle.dumps(BASE_VECTOR_STORE))
            SESSION_VECTOR_STORES[session_id] = cloned_store

        vector_store = SESSION_VECTOR_STORES[session_id]
        

        # Convert Pydantic model history to simple dictionaries
        history_dict = [msg.dict() for msg in request.conversation_history] if request.conversation_history else []

        # Generate response considering the history
        response = send_response(request.question, vector_store, history_dict)

        return {
            "answer": response,
            "message": "Response generated successfully"
        }
        
    except Exception as e:
        if session_id in SESSION_VECTOR_STORES:
            del SESSION_VECTOR_STORES[session_id]
        raise HTTPException(status_code=500, detail=str(e))

# Upload a user PDF file to update the vector store in memory
@app.post("/upload")
def upload_pdf(session_id: str = Body(...), file: UploadFile = File(...)):
    """
    En
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only allowed PDF files.")

    try:
        # If is the first time the session_id is used, create a new in-memory vector store
        if session_id not in SESSION_VECTOR_STORES:
            print(f"Creating new in-memory vector store for session: {session_id}")
            cloned_store = pickle.loads(pickle.dumps(BASE_VECTOR_STORE))
            SESSION_VECTOR_STORES[session_id] = cloned_store

        session_vector_store = SESSION_VECTOR_STORES[session_id]

        file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Update session vector store (in memory) with the new file
        update_in_memory_vector_store(file_path, session_vector_store)

        os.remove(file_path)

        return {"message": f"File processed in-memory for session {session_id}."}

    except Exception as e:
        # Clean up the session vector store if an error occurs
        if session_id in SESSION_VECTOR_STORES:
            del SESSION_VECTOR_STORES[session_id]
        raise HTTPException(status_code=500, detail=f"Error file processing: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running normally"
    }

# Endpoint to reset the vector index (useful in development)
@app.post("/reset-index")
def reset_vector_index():
    """
    Endpoint to recreate the vector index from scratch
    """
    try:
        create_index(force_recreate=True)
        return {"message": "Vector index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))