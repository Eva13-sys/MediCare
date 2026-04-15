def get_rag_response(msg, session_id, memory_manager, rag_chain):
    memory = memory_manager.get_or_create_memory(session_id)
    chat_history = memory_manager.get_conversation_history(session_id)

    history_text = ""
    for i, message in enumerate(chat_history[-6:]):
        role = "User" if i % 2 == 0 else "Assistant"
        history_text += f"{role}: {message.content}\n"

    response = rag_chain.invoke({
        "input": msg,
        "chat_history": history_text
    })

    return response