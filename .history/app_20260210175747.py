from flask import Flask, render_template, request, jsonify, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.memory_handler import memory_manager
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24) 

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

system_prompt_with_history=(
    "You are an medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the  "
    "answer concise."
    "\n\n"
    "Conversation History:\n{chat_history}\n\n"
    "Context:{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_with_history),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    print
    # input_text = msg
    print(f"User Question: {msg}")
    response = rag_chain.invoke({"input": msg})
    answer= response["answer"]

    sources= response.get("context",[])
    citations=[]

    for i, doc in enumerate(sources,1):
        source_file = doc.metadata.get("source", "Unknown Source")
        filename= source_file.split("\\")[-1].split("/")[-1]
        citations.append(f"[{i}]{filename}")

    if citations:
        formatted_response=f"{answer}\n\nSources:\n" + "\n".join(citations)
    else:
        formatted_response=answer

    print(f"Response: {formatted_response}")
    return str(formatted_response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
