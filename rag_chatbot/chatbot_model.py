import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_core.prompts import ChatPromptTemplate
system_prompt=ChatPromptTemplate.from_template("""
    You are a personal portfolio assistant chatbot for Gajraj, a Backend Developer.
    Your job is to answer questions ONLY using the information provided in the portfolio documents, resume, and knowledge base.
    STRICT RULES:

1. Only answer questions that are directly related to:
   - Gajraj's portfolio
   - Skills
   - Projects
   - Work experience
   - Education
   - Certifications
2. Use ONLY the provided context or documents to generate answers.
3. DO NOT:
   - Invent information
   - Use outside knowledge
   - Hallucinate answers
   - Generate assumptions
4. If the user's question is Not related to Gajraj Singh Rathore: Respond ONLY with:
"I am not able to respond to that. Please ask questions related to the Gajraj Singh Rathore."
5. Keep responses:
   - Professional
   - Based strictly on the provided context.

    <context>
    {context}
    </context>

    Question : {input}
""")

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=system_prompt
)

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

web_loader = WebBaseLoader("https://portfoliogajrajsinghrathore.netlify.app/")
web_docs = web_loader.load()

pdf_loader1 = PyPDFLoader(r"rag_chatbot/PersonalInformation.pdf")
pdf_doc1 = pdf_loader1.load()

pdf_loader = PyPDFLoader(r"rag_chatbot/GajrajSinghRathore.pdf")
pdf_doc = pdf_loader.load()

docs = pdf_doc1 + pdf_doc + web_docs

documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# SPLIT/CHUNK
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)  #500 character ka ek chunk and 200 character overlap
chunk_document = text_splitter.split_documents(documents)

# EMBEDDING/CONVERT INTO VECTORS  (VECTOR STORE)
db = FAISS.from_documents(chunk_document, HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2"))
db

retriever=db.as_retriever(search_kwargs={"k": 3}) 


from langchain_classic.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# answer = retrieval_chain.invoke({"input":question})
# answer["answer"]




# ---------------------------------------
# FINAL FUNCTION USED BY API
# ---------------------------------------
def ask_chatbot(question: str) -> str:
    """
    This function will be called by FastAPI.
    It receives a user question and returns the chatbot answer.
    """

    try:
        # 👉 CHANGE this to your final RAG pipeline
        response = retrieval_chain.invoke({"input": question})

        # Most RAG chains return answer inside "result" or "answer"
        return response.get("answer") or "No answer found"

    except Exception as e:
        return f"Error: {str(e)}"