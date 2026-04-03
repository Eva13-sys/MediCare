from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv() 

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()
index_name = "medicare"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(input_text)
    response = rag_chain.invoke({"input": msg})
    answer= response["answer"]

    sources= response.get("context",[])
    citations=[]

    for i, doc in enumerate(sources,1):
        source_file = doc.metadata.get("source", "Unknown Source")
        filename= source_file.split("\\")[-1].split("/")[-1]
        citations.append(f"filename)
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
