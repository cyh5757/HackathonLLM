from typing import Any, List

from pydantic import BaseModel
from langchain_core.documents import Document


class SimpleResponseMessage(BaseModel):
    message: str


class DocumentRes(BaseModel):
    page_content: str
    metadata: dict[str, Any]

    @classmethod
    def from_document(cls, doc: Document) -> "DocumentRes":
        return cls(
            page_content=doc.page_content,
            metadata=doc.metadata or {},
        )


class SseReq(BaseModel):
    query: str


class SnackRagDocument(BaseModel):
    index: int
    content: str
    metadata: dict[str, Any]
    score: float


class SnackContextPayload(BaseModel):
    status: str
    context: List[SnackRagDocument]
    found_docs: int


class SnackRagResponseChunk(BaseModel):
    status: str
    data: str
