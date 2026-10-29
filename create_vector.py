import os
import pandas as pd
from docx import Document
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
                
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        
    return text


def build_vector_store(file_path: str):

    pdf_text = extract_text_from_file(file_path)
    print("Reading and parsing document completed...")

    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-embedding-001"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    documents = text_splitter.split_text(pdf_text)

    vector_store = FAISS.from_texts(documents, embeddings)
    vector_store.save_local(folder_path="./vectorstore")
    print("Local Vector Database created successfully with gemini-embedding-001!")


if __name__ == "__main__":
    default_file = "./assets/uploaded_document.pdf"
    if os.path.exists(default_file):
        build_vector_store(default_file)