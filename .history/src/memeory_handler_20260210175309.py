from langchain.memory import ConversationBufferMemory
from typing import Dict
import uuid

class SessionMemoryManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferMemory] = {}
    
    def get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_conversation_history(self, session_id: str) -> list:
        if session_id in self.sessions:
            memory = self.sessions[session_id]
            return memory
        memory = self.get_or_create_memory(session_id)
        return memory.load_memory_variables({})["chat_history"]