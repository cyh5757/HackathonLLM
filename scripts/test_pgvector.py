import asyncio
from app.services.pgvector_service import search_pgvector
from app.api.deps import get_db  # get_db를 직접 호출

import sys
import asyncio

# Windows에서 psycopg + asyncio 문제 해결
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# 테스트용 질의어
test_query = "썬키스트의 첨가물을 보여줘"


# get_db는 async generator이므로 이렇게 사용
async def main():
    async for session in get_db():
        results = await search_pgvector(session=session, query=test_query)

        print(f"🔍 Query: {test_query}")
        for i, doc in enumerate(results):
            print(f"\n📄 Document {i + 1}:")
            print(f"Content: {doc.page_content}")  # ← 여기를 수정
            print(f"Metadata: {doc.metadata}")  # metadata는 그대로 사용 가능
            # print(f"Full: {doc.dict()}")


if __name__ == "__main__":
    asyncio.run(main())
