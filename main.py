from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import fitz
from transformers import pipeline
from supabase import create_client, Client
import shutil
import requests
from fastapi.exceptions import HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY =os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    pdf_id: str
    question: str

def extract_text_from_pdf_content(pdf_content: bytes) -> str:
    document = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

qa_pipeline = pipeline("text-generation", model="gpt2")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_content = await file.read()

    # Upload PDF to Supabase storage
    response = supabase.storage.from_("pdfs").upload(file.filename, file_content)
    
    if response.status_code != 200:
        return {"error": "Failed to upload PDF to Supabase storage"}
    
    return {"pdf_id": file.filename}

@app.post("/ask-question/")
async def ask_question(question: Question):
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/pdfs/{question.pdf_id}"

    response = requests.get(public_url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="PDF not found in Supabase storage")
    
    load_dotenv()
    TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    text = extract_text_from_pdf_content(response.content)


    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_texts(chunks, embeddings)    
    
    query = question.question
    if query:
        docs = db.similarity_search(query, k=5)
        llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5, "max_length":256})
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        return {"answer": response}
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
