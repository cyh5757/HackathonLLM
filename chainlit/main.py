import chainlit as cl
import traceback

import httpx
import json

# SSE용 엔드포인트
SSE_API_URL = "http://localhost:8000/api/v1/snacks/sse"
# RAG용 엔드포인트
RAG_API_URL = "http://localhost:8000/api/v1/snacks/test/rag"
# AGENT용 엔드포인트
AGENT_API_URL = "http://localhost:8000/api/v1/snacks/test/agent"

# Timeout 설정
custom_timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)


async def handle_sse_stream(query: str, streamed_message: cl.Message):
    """
    SSE API에 쿼리를 보내고 결과를 실시간 스트리밍.
    """
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            async with client.stream(
                "POST",
                SSE_API_URL,
                headers={"Content-Type": "application/json"},
                json={"query": query},
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
            error_detail = traceback.format_exc()
            await cl.Message(
                content=f"❌ SSE 요청 실패:\n```\n{error_detail}\n```"
            ).send()


async def handle_rag_query(query: str, streamed_message: cl.Message):
    """
    RAG API 호출
    """
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            response = await client.post(
                RAG_API_URL,
                headers={"Content-Type": "application/json"},
                json={"query": query},
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", "🤖 답변이 없습니다.")
                await streamed_message.stream_token(answer)
                await streamed_message.update()
            else:
                await cl.Message(content=f"❌ 서버 오류: {response.status_code}").send()

        except Exception as e:
            error_detail = traceback.format_exc()
            await cl.Message(content=f"❌ 요청 실패:\n```\n{error_detail}\n```").send()


async def handle_agent_query(query: str, streamed_message: cl.Message):
    """
    AGENT API 호출
    """
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            response = await client.post(
                AGENT_API_URL,
                headers={"Content-Type": "application/json"},
                json={"query": query},
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", "🤖 답변이 없습니다.")
                await streamed_message.stream_token(answer)
                await streamed_message.update()
            else:
                await cl.Message(
                    content=f"❌ AGENT 서버 오류: {response.status_code}"
                ).send()

        except Exception as e:
            error_detail = traceback.format_exc()
            await cl.Message(
                content=f"❌ AGENT 요청 실패:\n```\n{error_detail}\n```"
            ).send()


@cl.on_chat_start
async def start():
    await cl.Message(
        content="🤖 안녕하세요! 질문을 입력해주세요.\n\n"
        "- `rag:` 로 시작 → RAG 기반 응답\n"
        "- `agent:` 로 시작 → LangChain Agent 기반 응답\n"
        "- 그 외 → 기본 SSE 기반 응답"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    streamed_message = cl.Message(content="")
    await streamed_message.send()

    if query.lower().startswith("rag:"):
        await handle_rag_query(query[4:].strip(), streamed_message)
    elif query.lower().startswith("agent:"):
        await handle_agent_query(query[7:].strip(), streamed_message)
    else:
        await handle_sse_stream(query, streamed_message)
