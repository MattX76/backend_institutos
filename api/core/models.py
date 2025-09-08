from pydantic import BaseModel
from typing import List, Dict, Any

class UploadResponse(BaseModel):
    message: str
    document_id: str
    tenant_id: str

class QueryRequest(BaseModel):
    query: str
    tenant_id: str

class SourceNode(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceNode]