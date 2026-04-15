from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.helper import download_embeddings
import os

def get_retriever():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(host=os.environ.get("PINECONE_HOST"))

    embeddings = download_embeddings()

    docsearch = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    return docsearch.as_retriever(search_kwargs={"k": 7})