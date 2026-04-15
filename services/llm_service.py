from langchain_groq import ChatGroq
import os

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )