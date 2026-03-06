import os
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "portfolio-chatbot"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Prompt
system_prompt = ChatPromptTemplate.from_template("""
You are a personal portfolio assistant chatbot for Gajraj, a Backend Developer.

Your job is to answer questions ONLY using the information provided in the portfolio documents.

STRICT RULES:

1. Only answer questions related to:
   - Portfolio
   - Skills
   - Projects
   - Work experience
   - Education
   - Certifications

2. Use ONLY the provided context.

3. DO NOT:
   - Invent information
   - Use outside knowledge
   - Hallucinate answers

4. If the question is not related to Gajraj Singh Rathore respond ONLY with:

"I am not able to respond to that. Please ask questions related to the Gajraj Singh Rathore."

<context>
{context}
</context>

Question: {input}
""")

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Document chain
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=system_prompt
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load saved vector database
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# Function used by FastAPI
def ask_chatbot(question: str) -> str:
    try:
        response = retrieval_chain.invoke({"input": question})
        return response.get("answer") or "No answer found"

    except Exception as e:
        return f"Error: {str(e)}"
