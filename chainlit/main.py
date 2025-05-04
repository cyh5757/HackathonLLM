import chainlit as cl
import httpx
import json

# SSE용 엔드포인트
SSE_API_URL = "http://localhost:8000/api/v1/snacks/sse"

# RAG용 엔드포인트
RAG_API_URL = "http://localhost:8000/api/v1/snacks/test/rag"


async def handle_sse_stream(query: str, streamed_message: cl.Message):
    """
    주어진 쿼리를 SSE API에 전송하고 응답을 실시간으로 Chainlit 메시지에 스트리밍.
    """
    async with httpx.AsyncClient(timeout=None) as client:
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
            await cl.Message(content=f"❌ 서버 오류: `{str(e)}`").send()


async def handle_rag_query(query: str, streamed_message: cl.Message):
    """
    주어진 쿼리를 /test/rag API에 POST 요청하고 응답을 출력.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                RAG_API_URL,
                headers={"Content-Type": "application/json"},
                json={"query": query},
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "🤖 답변이 없습니다.")
                await streamed_message.stream_token(answer)
                await streamed_message.update()
            else:
                await cl.Message(content=f"❌ 서버 오류: {response.status_code}").send()

        except Exception as e:
            await cl.Message(content=f"❌ 요청 실패: `{str(e)}`").send()


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
