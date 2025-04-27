import chainlit as cl
import httpx
import json

API_URL = "http://localhost:8000/api/v1/snacks/sse"  # 실제 백엔드 URL로 수정


async def handle_sse_stream(query: str, streamed_message: cl.Message):
    """
    주어진 쿼리를 SSE API에 전송하고 응답을 실시간으로 Chainlit 메시지에 스트리밍.
    """
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
                        continue  # skip malformed data

                    status = data.get("status")
                    content = data.get("data", "")

                    if status == "processing" and content:
                        await streamed_message.stream_token(content)

                    elif status == "complete":
                        await streamed_message.update()
                        break

        except Exception as e:
            await cl.Message(content=f"❌ 서버 오류: `{str(e)}`").send()


@cl.on_chat_start
async def start():
    await cl.Message(content="🤖 안녕하세요! 질문을 입력하시면 바로 도와드릴게요.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    사용자의 메시지를 받아 SSE API에 전달하고 실시간 응답을 보여주는 핸들러.
    """
    # await cl.Message(content="🤖 질문을 처리 중입니다...").send()

    streamed_message = cl.Message(content="")  # 응답 누적 메시지
    await streamed_message.send()

    await handle_sse_stream(message.content, streamed_message)
