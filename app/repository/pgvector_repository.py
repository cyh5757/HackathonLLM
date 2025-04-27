import logging
import uuid
from datetime import datetime
from typing import Tuple, List

from app.models.models import LangchainPgEmbedding
from langchain_core.documents import Document
from sqlalchemy import update

from app.api.deps import SessionDep
from app.core import agent_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def search_similar_memories(
    query: str,
    key_prefix: str,
    field_name: str,
    size: int
) -> List[Tuple[Document, float]]:
    embedding_result = await agent_tools.embeddings.aembed_documents([query])
    query_vector: list[float] = embedding_result[0]

    # TODO
    # filter_clause = {"prefix": {"$eq": key_prefix}, "field_name": {"$eq": field_name}}

    results: List[Tuple[Document, float]] = await agent_tools.vector_store.asimilarity_search_with_score_by_vector(
        embedding=query_vector,
        k=size,
        filter={}
    )
    logger.info(
        f"\n[_search_similar_memories]Search similar memories \n{[(x[0].id, x[0].page_content) for x in results]}\n")
    return results


async def update_memory(
    *,
    db_session: SessionDep,
    target_memory_list: List[Tuple[Document, float]],
    new_memory: Document,
    key_prefix: str,
    field_name: str,
    memory_type: str,
):
    if not target_memory_list or not new_memory:
        return
    doc = new_memory.model_copy()
    doc.metadata.update({"prefix": key_prefix, "field_name": field_name})

    ids: list[str] = [str(m[0].id) for m in target_memory_list]

    for memory_id in ids:
        stmt = (
            update(LangchainPgEmbedding)
            .where(LangchainPgEmbedding.id == memory_id)
            .values(
                cmetadata=doc.metadata,
                modified_at=datetime.now(),
                modified_by=key_prefix
            )
        )
        await db_session.execute(stmt)

    logger.info(
        f"\n[_update_memory] Updated memory_type={memory_type}, prefix={key_prefix}, ids={ids}\n{new_memory.metadata}\n")


async def insert_memory(
    *,
    db_session: SessionDep,
    new_memory: Document,
    key_prefix: str,
    field_name: str,
):
    new_key = str(uuid.uuid4())

    doc = new_memory.model_copy()
    doc.metadata.update({"prefix": key_prefix, "field_name": field_name})

    embedding_result = await agent_tools.embeddings.aembed_documents([doc.page_content])
    query_vector: list[float] = embedding_result[0]

    await agent_tools.vector_store.aadd_embeddings(
        texts=[doc.page_content],
        embeddings=[query_vector],
        metadatas=[doc.metadata],
        ids=[new_key]
    )
    logger.info(f"\n[_insert_memory] Inserted prefix={key_prefix}, id={new_key}\n{doc}\n")
