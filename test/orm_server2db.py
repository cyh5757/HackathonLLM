from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL is not None, "DATABASE_URL 환경 변수가 설정되지 않았습니다."
print(DATABASE_URL)
# DATABASE_URL = "postgresql+asyncpg://postgres:123123@localhost:5432/test"

engine = create_async_engine(DATABASE_URL, echo=True)


async def test_connection():
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        print("DB 연결 성공, 결과:", result.scalar())


if __name__ == "__main__":
    asyncio.run(test_connection())
