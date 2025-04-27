import json
from typing import AsyncGenerator, Any

from app.core import agent_tools


async def sse_stream_generator(data_gen: AsyncGenerator[Any, None]) -> AsyncGenerator[str, None]:
    async for chunk in data_gen:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

# 단순 챗봇 답변 생성에 사용
async def stream_llm_response(prompt: str) -> AsyncGenerator[dict, None]:
    async for chunk in agent_tools.llm.astream(prompt):
        content = getattr(chunk, "content", "")
        if content:
            yield {"status": "processing", "data": content}
    yield {"status": "complete", "data": "Stream finished"}
