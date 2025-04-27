import logging
from datetime import datetime
from typing import AsyncGenerator, Any
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.api.deps import SessionDep
from app.api.dto.models import SnackContextPayload, SnackRagResponseChunk, SnackRagDocument
from app.core import agent_tools
from app.repository import pgvector_repository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SNACK_RAG_PROMPT = """
당신은 과자와 식품 첨가물에 대한 정보를 제공하는 도우미입니다.
아래 정보를 바탕으로 사용자의 질문에 답변해주세요.

오늘 날짜와 시간: {datetime}

사용자 질문: {query}

관련 정보:
{context}

답변할 때 다음 사항을 지켜주세요:
1. 주어진 정보만을 활용하여 답변하세요.
2. 정보가 부족하면 솔직하게 모른다고 답변하세요.
3. 과학적 사실에 기반하여 정확한
 정보를 제공하세요.
4. 사용자가 특정 과자나 첨가물에 대해 물어보면 해당 정보에 집중하여 답변하세요.
5. 알레르기 정보나 안전 관련 정보가 있다면 반드시 언급해주세요.
"""


async def generate_rag_response(
    session: SessionDep,
    query: str,
):
    """
    주어진 쿼리에 대해 RAG(Retrieval-Augmented Generation) 응답을 생성합니다.
    
    1. 벡터 DB에서 쿼리와 관련된 과자 및 첨가물 정보를 검색합니다.
    2. 검색된 정보를 컨텍스트로 LLM에 질의합니다.
    3. LLM 응답을 반환합니다.
    """
    # 1. 벡터 DB에서 쿼리와 관련된 정보 검색
    similar_docs: list[tuple[Document, float]] = await pgvector_repository.search_similar_memories(
        query=query,
        key_prefix="snack",  # 적절한 키 프리픽스로 변경 필요
        field_name="snack_data",  # 적절한 필드 이름으로 변경 필요
        size=10  # 검색할 문서 수
    )

    # 검색 결과가 없는 경우
    if not similar_docs:
        return "검색 결과가 없습니다. 다른 질문을 해보세요."

    # 2. 컨텍스트 구성
    context = "\n\n".join([
        f"문서 {i + 1}:\n{doc.page_content}\n메타데이터: {doc.metadata}"
        for i, (doc, _) in enumerate(similar_docs)
    ])

    # 3. LLM에 질의
    prompt = ChatPromptTemplate.from_template(SNACK_RAG_PROMPT)
    chain = prompt | agent_tools.llm

    response = await chain.ainvoke({
        "datetime": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y년 %m월 %d일 %H시 %M분"),
        "query": query,
        "context": context
    })

    return response.content


async def get_rag_response_with_context(
    session: SessionDep,
    query: str,
) -> AsyncGenerator[dict, None]:
    similar_docs: list[tuple[Document, float]] = await pgvector_repository.search_similar_memories(
        query=query,
        key_prefix="snack",
        field_name="snack_data",
        size=10
    )

    if not similar_docs:
        yield SnackContextPayload(
            status="complete",
            context=[],
            found_docs=0
        ).model_dump()
        yield SnackRagResponseChunk(
            status="complete",
            data="검색 결과가 없습니다. 다른 질문을 해보세요."
        ).model_dump()
        return

    context_list = [
        SnackRagDocument(
            index=i + 1,
            content=doc.page_content,
            metadata=doc.metadata,
            score=score
        )
        for i, (doc, score) in enumerate(similar_docs)
    ]

    # LLM용 텍스트 context
    context_str = "\n\n".join([
        f"문서 {doc.index}:\n{doc.content}\n메타데이터: {doc.metadata}"
        for doc in context_list
    ])

    prompt = ChatPromptTemplate.from_template(SNACK_RAG_PROMPT)
    chain = prompt | agent_tools.llm

    formatted_input = {
        "datetime": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y년 %m월 %d일 %H시 %M분"),
        "query": query,
        "context": context_str
    }

    yield SnackContextPayload(
        status="context",
        context=context_list,
        found_docs=len(similar_docs)
    ).model_dump()

    async for chunk in chain.astream(formatted_input):
        content = getattr(chunk, "content", "")
        if content:
            yield SnackRagResponseChunk(
                status="processing",
                data=content
            ).model_dump()

    yield SnackRagResponseChunk(
        status="complete",
        data="RAG 응답이 완료되었습니다."
    ).model_dump()


async def get_context_only(
    session: SessionDep,
    query: str,
) -> dict[str, Any]:
    similar_docs: list[tuple[Document, float]] = await pgvector_repository.search_similar_memories(
        query=query,
        key_prefix="snack",
        field_name="snack_data",
        size=10
    )

    if not similar_docs:
        return SnackContextPayload(
            status="complete",
            context=[],
            found_docs=0
        ).model_dump()

    context_list = [
        SnackRagDocument(
            index=i + 1,
            content=doc.page_content,
            metadata=doc.metadata,
            score=score
        )
        for i, (doc, score) in enumerate(similar_docs)
    ]

    return SnackContextPayload(
        status="context",
        context=context_list,
        found_docs=len(similar_docs)
    ).model_dump()
