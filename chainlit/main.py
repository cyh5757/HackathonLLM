import chainlit as cl
import traceback

import httpx
import json

# 2375 암호화 x
# 2376 암호화 o
# 80 8080 http
# 443 https
# chainlit run chainlit/main.py --port 8501

# SSE용 엔드포인트
SSE_API_URL = "http://localhost:8000/api/v1/snacks/sse"
# RAG용 엔드포인트
RAG_API_URL = "http://localhost:8000/api/v1/snacks/test/rag"

# Timeout 설정 (connect, read, write, pool 각각)
custom_timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)


async def handle_sse_stream(query: str, streamed_message: cl.Message):
    """
    주어진 쿼리를 SSE API에 전송하고 응답을 실시간으로 Chainlit 메시지에 스트리밍.
    """
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            async with client.stream(
                "POST",
                SSE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
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


@cl.on_chat_start
async def start():
    await cl.Message(
        content="🤖 안녕하세요! 질문을 입력하시면 도와드릴게요.\n\n- `rag:`로 시작하면 RAG 기반 응답\n- 그 외는 SSE 기반 응답"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    streamed_message = cl.Message(content="")
    await streamed_message.send()

    if query.lower().startswith("rag:"):
        pure_query = query[4:].strip()
        await handle_rag_query(pure_query, streamed_message)
    else:
        await handle_sse_stream(query, streamed_message)
