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
                out
            )
        return self.sessions[session_id]