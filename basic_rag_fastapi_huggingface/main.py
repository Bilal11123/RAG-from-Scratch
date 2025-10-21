from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import create_vectorstore, create_rag_chain
import os

app = FastAPI(title="LangChain RAG API")

# Initialize DB if not exists
if not os.path.exists("chroma_db"):
    print("Creating vectorstore...")
    create_vectorstore()

rag_chain = create_rag_chain()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    response = rag_chain.run(request.query)
    return {"query": request.query, "answer": response}
