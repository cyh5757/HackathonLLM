# scripts/load_snack_to_vectordb.py
import asyncio
from app.api.deps import get_db
from app.services.load_snack_service import load_db_data_to_vectordb  # 너가 정리한 로딩 서비스 import
from sqlalchemy.ext.asyncio import AsyncSession

async def main():
    # Session 만들기
    async for session in get_db():
        await load_db_data_to_vectordb(session, batch_size=100)  # 100개만 넣기
        break

if __name__ == "__main__":
    asyncio.run(main())
