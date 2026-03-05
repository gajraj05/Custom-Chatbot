# import logging
# from fastapi import FastAPI
# import inngest
# import inngest.fast_api
# from inngest.experimental import ai
# from dotenv import load_dotenv
# import os
# import uuid
# import datetime



# load_dotenv()

# inngest_client = inngest.Inngest(
#     app_id="rag_app",
#     logger=logging.getLogger("unicorn"),
#     is_production=False,
#     serializer=inngest.PydanticSerializer(),
# )

# @inngest_client.create_function(
#     fn_id="RAG: Ingest PDF",
#     trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
# )

# async def rag_ingest_pdf(ctx: inngest.Context):
#     return {"Hello": "World"}


# app = FastAPI()

# inngest_app = inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf])

# # if __name__ == "__main__":
# #     main()

# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from rag_chatbot.chatbot_model import ask_chatbot

print("Starting FastAPI server...")

app = FastAPI()

# Request body
class Query(BaseModel):
    question: str

# API Endpoint
@app.post("/chat")
def chat(query: Query):
    answer = ask_chatbot(query.question)
    return {"answer": answer}
