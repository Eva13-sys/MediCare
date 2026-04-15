from flask import Flask, render_template, request, jsonify, session
from sqlalchemy import text
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from src.memory_handler import memory_manager
from src.confidence_scorer import extract_confidence, get_confidence_label
from agents.rag_agent import get_rag_response
from agents.summarizer_agent import summarize
from agents.validator_agent import validate
from agents.advisor_agent import get_advice
from metrics import get_metrics
from metrics import log_request

import os
import uuid
import time

from pinecone import Pinecone

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()

# ENV
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
PINECONE_HOST = os.environ.get('PINECONE_HOST')

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")
if not PINECONE_HOST:
    raise ValueError("Missing PINECONE_HOST")

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

embeddings = download_embeddings()

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')


# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form["msg"]

#     session_id = session.get('session_id', str(uuid.uuid4()))
#     session['session_id'] = session_id

#     memory = memory_manager.get_or_create_memory(session_id)
#     chat_history = memory_manager.get_conversation_history(session_id)

#     # format history
#     history_text = ""
#     for i, message in enumerate(chat_history[-6:]):
#         role = "User" if i % 2 == 0 else "Assistant"
#         history_text += f"{role}: {message.content}\n"

#     # invoke RAG
#     response = rag_chain.invoke({
#         "input": msg,
#         "chat_history": history_text
#     })

#     raw_answer = response["answer"]

#     # confidence parsing
#     clean_answer, confidence_score = extract_confidence(raw_answer)
#     confidence_label = get_confidence_label(confidence_score)

#     # save memory
#     memory.chat_memory.add_user_message(msg)
#     memory.chat_memory.add_ai_message(clean_answer)

#     sources = response.get("context", [])
#     citations_set = set()

#     for i, doc in enumerate(sources, 1):
#         source_file = doc.metadata.get("source", "Unknown")
#         filename = source_file.split("\\")[-1].split("/")[-1]
#         citations_set.add(f"{filename} (chunk {i})")

#     citations = list(citations_set)

#     # format response
#     formatted_response = clean_answer
#     formatted_response += f"\n\nConfidence: {confidence_label} ({confidence_score}%)"

#     if citations:
#         formatted_response += "\n\nSources:\n"
#         for i, src in enumerate(citations, 1):
#             formatted_response += f"[{i}] {src}\n"

#     return formatted_response

@app.route("/get", methods=["POST"])
def chat():
    start_time = time.time()

    msg = request.form["msg"]

    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id

    response = get_rag_response(msg, session_id, memory_manager, rag_chain)

    raw_answer = response["answer"]

    clean_answer, confidence_score = extract_confidence(raw_answer)
    confidence_label = get_confidence_label(confidence_score)

    latency = round((time.time() - start_time) * 1000, 2)
    log_request(latency, "/get")

    return jsonify({
        "answer": clean_answer,
        "confidence": confidence_score,
        "latency_ms": latency
    })

@app.route("/ask-doc", methods=["POST"])
def ask_doc():
    start_time = time.time()

    query = request.json.get("query")

    response = rag_chain.invoke({
        "input": query,
        "chat_history": ""
    })

    latency = round((time.time() - start_time) * 1000, 2)
    log_request(latency, "/ask-doc")

    return jsonify({
        "answer": response.get("answer", ""),
        "latency_ms": latency
    })

@app.route("/summarize", methods=["POST"])
def summarize_route():
    text = request.json.get("text")
    start_time = time.time()
    summary = summarize(chatModel, text)
    latency = round((time.time() - start_time) * 1000, 2)
    log_request(latency, "/summarize")
    return jsonify({"summary": summary, "latency_ms": latency})

@app.route("/validate", methods=["POST"])
def validate_route():
    answer = request.json.get("answer")
    start_time = time.time()
    is_valid = validate(answer)
    latency = round((time.time() - start_time) * 1000, 2)
    log_request(latency, "/validate")
    return jsonify({"is_valid": is_valid, "latency_ms": latency})

@app.route("/advice", methods=["POST"])
def advice():
    symptoms = request.json.get("symptoms")
    start_time = time.time()
    advice = get_advice(chatModel, symptoms)
    latency = round((time.time() - start_time) * 1000, 2)
    log_request(latency, "/advice")
    return jsonify({"advice": advice, "latency_ms": latency})


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(get_metrics())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/clear", methods=["POST"])
def clear_chat():
    session_id = session.get('session_id')
    if session_id:
        memory_manager.clear_session(session_id)
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)