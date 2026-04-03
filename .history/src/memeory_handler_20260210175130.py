from langchain.memory import ConversationBufferMemory
from typing import Dict
import uuid

class SessionMemoryManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferMemory] = {}
    
    def get_or_create_memory(self, session_id)