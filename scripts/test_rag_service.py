import asyncio
from app.api.deps import get_db
from app.services.rag_service_test import generate_rag_response

import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

query = "난 딸기 맛을 좋아해. 알레르기가 없는 과자 중에 추천해줘 "  # 또는 "쫀디기의 위험한 성분이 있나요?" 등 테스트용, 썬키스트의 첨가물을 알려줘


async def main():
    async for session in get_db():
        response = await generate_rag_response(session=session, query=query)
        print(f"\n🔍 Query: {query}")
        print(f"\n📘 RAG Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
