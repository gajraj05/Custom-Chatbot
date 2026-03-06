import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from rag_chatbot.chatbot_model import ask_chatbot

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Portfolio Chatbot API Running"}

@app.post("/chat")
def chat(query: Query):
    answer = ask_chatbot(query.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
