from flask import Blueprint, request, jsonify, session
import uuid

rag_bp = Blueprint("rag", __name__)

def init_rag_routes(rag_chain, memory_manager):
    
    @rag_bp.route("/get", methods=["POST"])
    def chat():
        msg = request.form["msg"]

        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id

        response = rag_chain.invoke({
            "input": msg,
            "chat_history": ""
        })

        return jsonify({"answer": response["answer"]})

    return rag_bp