from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from generative_resp.ai_response import send_response, create_index, update_vector_store_with_new_file

import shutil
import os
import uuid

app = FastAPI()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Modelos Pydantic para las requests
class Message(BaseModel):
    role: str  # "human" o "ai"
    content: str

class QuestionRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Message]] = []

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint para hacer preguntas con historial de conversación opcional
    
    El historial se pasa en cada request desde el frontend y no se almacena en el servidor
    """
    try:
        # Convertir el historial de Pydantic models a diccionarios simples
        history_dict = []
        if request.conversation_history:
            history_dict = [
                {"role": msg.role, "content": msg.content} 
                for msg in request.conversation_history
            ]
        
        # Generar respuesta considerando el historial
        response = send_response(request.question, history_dict)
        
        return {
            "answer": response,
            "message": "Response generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint para subir archivos PDF (sin cambios en la funcionalidad)
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only allowed PDF files.")

    try:
        # Save the uploaded file to a temporary location
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(TEMP_DIR, unique_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load the existing vector store or create a new one
        vector_store = create_index(force_recreate=False)

        # Update the vector store with the new file
        update_vector_store_with_new_file(file_path, vector_store)

        # Remove the temporary file
        os.remove(file_path)

        return {"message": "File loaded and vectorial DB updated successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error file processing: {str(e)}")

# Endpoint de salud/estado
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running normally"
    }

# Endpoint para limpiar el índice vectorial (útil para desarrollo)
@app.post("/reset-index")
def reset_vector_index():
    """
    Endpoint para recrear el índice vectorial desde cero
    """
    try:
        create_index(force_recreate=True)
        return {"message": "Vector index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))