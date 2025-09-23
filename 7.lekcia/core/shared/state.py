from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    # full conversation
    messages: List[BaseMessage]
    # optional structured fields
    nextStep: str