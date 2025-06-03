from fastapi import FastAPI, HTTPException, UploadFile, File
from generative_resp.ai_response import send_response, create_index, update_vector_store_with_new_file

import shutil
import os
import uuid

app = FastAPI()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/ask")
def ask_question(question: str):
    try:
        response = send_response(question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
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
