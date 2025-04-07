from fastapi import FastAPI, UploadFile, HTTPException, Request, Response
import shutil
from pydantic import BaseModel
import os
import boto3
import logging
import tempfile
from typing import Optional, Union
from fastapi.responses import JSONResponse
from ingest import ingest_documents, create_vector_store_with_retry, create_or_load_faiss
from rag_bot import create_rag_bot, ask_question, load_csv_from_s3, session_memory
from config import S3_BUCKET_NAME
import uuid

class QuestionRequest(BaseModel):
    question: str  # ✅ No session_id required

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Ask your question here "
            }
        }

class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[list[str]]=None

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")
vector_store = None
file_urls = {}

@app.on_event("startup")
async def startup_event():
    global vector_store, file_urls
    try:
        vector_store = create_or_load_faiss()
        print("✅ Loaded existing vector store from S3")

        # ✅ Load file URLs from S3
        file_urls = load_csv_from_s3(S3_BUCKET_NAME, "Urls.csv")
        if file_urls is None:
            file_urls = {}

    except Exception as e:
        print(f"⚠️ No FAISS index found in S3: {e}. Upload a document first.")
        vector_store = None
        file_urls = {}

@app.post("/upload/")
async def upload_document(file: UploadFile):
    global vector_store
    try:
        temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Uploading {file.filename} to S3...")
        with open(temp_file_path, "rb") as buffer:
            s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, file.filename)
        logger.info("✅ Upload successful!")

        logger.info(f"Processing file {file.filename} for vector store...")
        chunks = ingest_documents(temp_file_path)
        vector_store = create_vector_store_with_retry(chunks, os.getenv("OPENAI_API_KEY"))
        vector_store.save_local("faiss_index")  

        return {"message": "File uploaded & indexed successfully!"}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return {"message": f"Failed to upload document: {e}"}

@app.post("/ask/", response_model=QuestionResponse)
async def ask(request: QuestionRequest, req: Request, res: Response):
    """
    Ask a question while maintaining short-term memory **automatically** (like ChatGPT).
    """
    global vector_store, file_urls

    # ✅ Auto-generate or retrieve session ID internally
    session_id = req.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        res.set_cookie(key="session_id", value=session_id, httponly=True)

    if vector_store is None:
        try:
            vector_store = create_or_load_faiss()
            if vector_store is None:
                raise HTTPException(
                    status_code=400,
                    detail="⚠️ No documents available. Please upload a document first."
                )
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="⚠️ No documents available. Please upload a document first."
            )

    try:
        qa_chain = create_rag_bot(vector_store)
        
        # ✅ FIX: Pass `file_urls` when calling `ask_question()`
        response = ask_question(qa_chain, request.question, session_id, file_urls)
        # Ensure we always return a proper JSON structure
        if isinstance(response, dict):
            if "sources" not in response:
                return JSONResponse(content={"answer": response["answer"]})
            return response
        else:
            # Handle unexpected string responses (shouldn't happen with above fixes)
            return JSONResponse(content={"answer": str(response)})
    except Exception as e:
        logger.error(f"⚠️ Internal Server Error: {e}")
        return JSONResponse(
            content={"answer": f"⚠️ Internal Server Error: {str(e)}"},
            status_code=500
        )