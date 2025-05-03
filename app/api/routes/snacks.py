import json
import logging

from fastapi import APIRouter
from fastapi import Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import SessionDep
from app.api.dto.models import SimpleResponseMessage, SseReq
from app.api.routes.util import sse_stream_generator
from app.core import agent_tools
from app.models import snack_types
from app.models.snack_dto import SnackDetailDto
from app.repository import snack_repository
from app.repository.snack_vector_loader import load_db_data_to_vectordb
from app.services import rag_service_test  # 테스트용 서비스 추가
from app.services.rag_service_test import get_rag_response_with_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/snacks", tags=["memories"])


@router.get("/test")
async def get_snack_test(
    *, session: SessionDep, response: Response
) -> SimpleResponseMessage:
    response.status_code = status.HTTP_200_OK

    result: list[SnackDetailDto] = (
        await snack_repository.get_snacks_without_allergy_keyword(
            session, snack_types.AllergyKeyword.대두
        )
    )

    await session.commit()

    return SimpleResponseMessage(message=f"HI {str(result)}")


@router.post("/sse")
async def ask_query(
    *,
    request: SseReq,
    session: SessionDep,
) -> StreamingResponse:
    query: str = request.query

    async def stream_llm_response(prompt: str):
        async for chunk in agent_tools.llm.astream(
            prompt
        ):  # 비동기 제너레이터라고 가정
            content = getattr(chunk, "content", "")
            if content:
                yield f"data: {json.dumps({'status': 'processing', 'data': content}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'status': 'complete', 'data': 'Stream finished'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_llm_response(query), media_type="text/event-stream")


@router.post("/update-vector-db")
async def update_vector_db(session: SessionDep) -> SimpleResponseMessage:
    count = await load_db_data_to_vectordb(session)
    return SimpleResponseMessage(
        message=f"{count}개의 문서가 벡터 DB에 저장되었습니다."
    )


# 테스트용 엔드포인트
@router.post("/test/rag")
async def test_rag_query(
    *,
    request: SseReq,
    session: SessionDep,
) -> SimpleResponseMessage:
    """
    테스트용 RAG 기반 질의응답 API
    """
    query: str = request.query
    response_text = await rag_service_test.generate_rag_response(
        session=session, query=query
    )

    return SimpleResponseMessage(message=response_text)


@router.post("/test/rag/context")
async def test_rag_query_with_context(
    *,
    request: SseReq,
    session: SessionDep,
):
    return StreamingResponse(
        sse_stream_generator(get_rag_response_with_context(session, request.query)),
        media_type="text/event-stream",
    )


@router.post("/test/rag/context-only")
async def test_context_only_query(
    *,
    request: SseReq,
    session: SessionDep,
):
    """
    테스트용 컨텍스트만 반환하는 API (LLM 호출 없음)
    """
    query: str = request.query
    result = await rag_service_test.get_context_only(session=session, query=query)

    return result
