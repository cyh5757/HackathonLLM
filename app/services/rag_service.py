import logging
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

# from app.service.generized_metadata import GenerizedMetadata
from app.api.deps import SessionDep
from app.repository import pgvector_repository
from langchain_core.documents import Document
from app.repository import snack_repository
# from app.schemas.snack import SnackDetailDto
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_context_only(
    session: SessionDep,
    query: str,
    key_prefix: str = "snack",
    field_name: str = "snack_data",
    size: int = 10
):
    """
    쿼리와 관련된 컨텍스트만 검색하여 반환합니다. LLM은 호출하지 않습니다.
    
    Args:
        session: 데이터베이스 세션
        query: 검색 쿼리
        key_prefix: 벡터 저장소의 키 프리픽스
        field_name: 벡터 저장소의 필드 이름
        size: 검색할 문서 수
        
    Returns:
        컨텍스트 및 문서 정보를 포함한 딕셔너리
    """
    # 벡터 DB에서 쿼리와 관련된 정보 검색
    similar_docs: list[tuple[Document, float]] = await pgvector_repository.search_similar_memories(
        query=query,
        key_prefix=key_prefix,
        field_name=field_name,
        size=size
    )
    # 검색 결과가 없는 경우
    if not similar_docs:
        return {
            "context": "",
            "found_docs": 0,
            "documents": []
        }


    # 컨텍스트 구성
    context = "\n\n".join([
        f"문서 {i+1}:\n{doc.page_content}\n메타데이터: {doc.metadata}"
        for i, (doc, _) in enumerate(similar_docs)
    ])
    
    # 개별 문서 정보도 추가
    documents = []
    for i, (doc, score) in enumerate(similar_docs):
        
        documents.append({
            "index": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    
    return {
        "context": context,
        "found_docs": len(similar_docs),
        "documents": documents
    } 

# async def ID_search(
#     db_session: SessionDep,
#     doc: Document
# ) -> Optional[SnackDetailDto]:
#     """
#     벡터 검색 결과의 문서(doc)에서 메타데이터에 포함된 ID를 기반으로,
#     데이터베이스에서 상세 스낵 정보를 조회합니다.
#     """
#     snack_id = doc.metadata.get("id")

#     if snack_id is None:
#         raise ValueError("문서 메타데이터에 'id'가 없습니다.")

#     snack_detail = await snack_repository.get_snack_detail_by_id(
#         db_session=db_session,
#         snack_id=snack_id
#     )
#     return snack_detail


    