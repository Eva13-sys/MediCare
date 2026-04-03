from langchain.memory import ConversationBufferMemory
from typing import Dict
import uuid

class SessionMemoryManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferMemory] = {}