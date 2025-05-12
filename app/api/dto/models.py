from typing import Any, List

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from typing import TypedDict


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


class Decision_maker(BaseModel):
    reference: str = Field(
        description="Choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material."
    )


class GraphState(TypedDict):
    question: str
    context: list | str
    organize_reference: str
