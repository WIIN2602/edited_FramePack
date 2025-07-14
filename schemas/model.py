from typing import List
from pydantic import BaseModel
    
class SceneUploadEntry(BaseModel):
    uid: str
    source_uid: str
    timestamp: List[float] = []
    duration: float = 0.0
    caption: str = ""
    embedding: List[float] = []
    session_id: str = ""