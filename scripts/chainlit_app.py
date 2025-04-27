# scripts/chainlit_app.py

import sys
import chainlit as cl
import asyncio
import httpx
import json
from app.api.deps import get_db
from app.services.rag_service import ID_search
from app.services.pgvector_service import search_pgvector  
from langchain_core.documents import Document

# Windows에서는 asyncio 정책 강제 세팅
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

API_URL = "http://localhost:8000/api/v1/snacks/sse"

# FastAPI SSE API를 이용해서 답변 받기
async def handle_sse_stream(query: str, streamed_message: cl.Message):
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                API_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"query": query}
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    data_str = line.removeprefix("data:").strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    status = data.get("status")
                    content = data.get("data", "")

                    if status == "processing" and content:
                        await streamed_message.stream_token(content)

                    elif status == "complete":
                        await streamed_message.update()
                        break

        except Exception as e:
            await cl.Message(content=f"❌ 서버 오류: `{str(e)}`").send()

# 채팅 시작 시
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="🤖 안녕하세요! 질문을 입력해주세요!").send()

# 사용자 입력 처리
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content
    streamed_message = cl.Message(content="")
    await streamed_message.send()

    # 세션 열기
    db_gen = get_db()
    session = await anext(db_gen)

    try:
        # 🔥 Step1: 벡터 검색 + LLM 기반 결과 선별
        filtered_documents = await search_pgvector(session=session, query=query)

        if not filtered_documents:
            await streamed_message.stream_token("❗ 관련된 스낵 정보를 찾지 못했습니다.\n\n")
        else:
            for doc_res in filtered_documents:
                doc = Document(page_content=doc_res.page_content, metadata=doc_res.metadata)

                # 🔥 Step2: ID로 스낵 상세 조회
                snack_detail = await ID_search(db_session=session, doc=doc)

                if snack_detail:
                    snack_info = (
                        f"✅ [스낵 이름] {snack_detail.name}\n"
                        f"🏢 [제조사] {snack_detail.company}\n"
                        f"📦 [바코드] {snack_detail.barcode}\n"
                        f"🌟 [스낵 종류] {snack_detail.snack_type}\n"
                        f"🛡️ [안전마크] {', '.join(snack_detail.safe_food_mark_list)}\n\n"
                    )
                    await streamed_message.stream_token(snack_info)

        # 🔥 Step3: FastAPI SSE 호출
        await handle_sse_stream(query, streamed_message)

    except Exception as e:
        await streamed_message.stream_token(f"❌ 오류 발생: {str(e)}")
    finally:
        await streamed_message.update()
        await session.aclose()
