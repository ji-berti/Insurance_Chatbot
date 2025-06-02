# from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from generative_resp.ai_response import send_response

app = FastAPI()

@app.post("/ask")
def ask_question(question: str):
    try:
        response = send_response(question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_pdf():
    try:
        # ...
        return {"message": "Archivo subido y base de datos actualizada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))