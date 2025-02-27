# main.py
from fastapi import FastAPI, UploadFile, Request, HTTPException
import shutil
from pydantic import BaseModel
import os
import boto3
import logging
import tempfile
from ingest import ingest_documents, create_vector_store_with_retry, create_or_load_faiss
from rag_bot import create_rag_bot, ask_question, load_csv_from_s3
from config import S3_BUCKET_NAME

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Your Question here"
            }
        }

class KeywordClarificationRequest(BaseModel):
    question: str
    keywords: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    needs_clarification: bool = False

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")
vector_store = None

file_urls = load_csv_from_s3(S3_BUCKET_NAME, "Urls.csv")

@app.on_event("startup")
async def startup_event():
    global vector_store
    try:
        # Load the vector store directly, not as a retriever
        vector_store = create_or_load_faiss()
        print("✅ Loaded existing vector store from S3")
    except Exception as e:
        print(f"⚠️ No FAISS index found in S3: {e}. Upload a document first.")
        vector_store = None

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
        vector_store.save_local("faiss_index")  # Save to avoid reloading issues


        return {"message": "File uploaded & indexed successfully!", "index_name": "latest.index"}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return {"message": f"Failed to upload document: {e}"}

@app.post("/ask/", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """
    Ask a question about the uploaded documents.
    """
    global vector_store

    if vector_store is None:
        try:
            # Try loading vector store again
            vector_store = create_or_load_faiss()
            if vector_store is None:
                raise HTTPException(
                    status_code=400, 
                    detail="⚠️ No documents available. Please upload a document first."
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="⚠️ No documents available. Please upload a document first."
            )

    try:
        # Create RAG bot with the vector store
        qa_chain = create_rag_bot(vector_store)
        answer, sources, needs_clarification_flag = ask_question(qa_chain, request.question, file_urls)
        
        return QuestionResponse(
            answer=answer, 
            sources=sources,
            needs_clarification=needs_clarification_flag
        )
    
    except Exception as e:
        logger.error(f"⚠️ Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"⚠️ Internal Server Error: {str(e)}")

@app.post("/ask_with_keywords/", response_model=QuestionResponse)
async def ask_with_keywords(request: KeywordClarificationRequest):
    """
    Ask a question with additional clarifying keywords.
    """
    global vector_store

    if vector_store is None:
        try:
            vector_store = create_or_load_faiss()
            if vector_store is None:
                raise HTTPException(
                    status_code=400, 
                    detail="⚠️ No documents available. Please upload a document first."
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="⚠️ No documents available. Please upload a document first."
            )

    try:
        # Create RAG bot with the vector store
        qa_chain = create_rag_bot(vector_store)
        answer, sources, needs_clarification_flag = ask_question(
            qa_chain, 
            request.question, 
            file_urls,
            keywords=request.keywords
        )
        
        return QuestionResponse(
            answer=answer, 
            sources=sources,
            needs_clarification=needs_clarification_flag
        )
    
    except Exception as e:
        logger.error(f"⚠️ Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=f"⚠️ Internal Server Error: {str(e)}")