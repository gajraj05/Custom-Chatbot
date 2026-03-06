import os
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "portfolio-chatbot"

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load documents
web_loader = WebBaseLoader("https://portfoliogajrajsinghrathore.netlify.app/")
web_docs = web_loader.load()

pdf_loader1 = PyPDFLoader("rag_chatbot/PersonalInformation.pdf")
pdf_doc1 = pdf_loader1.load()

pdf_loader2 = PyPDFLoader("rag_chatbot/GajrajSinghRathore.pdf")
pdf_doc2 = pdf_loader2.load()

docs = web_docs + pdf_doc1 + pdf_doc2

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

documents = text_splitter.split_documents(docs)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = FAISS.from_documents(documents, embeddings)

# Save DB
db.save_local("vector_db")

print("Vector database created successfully.")
