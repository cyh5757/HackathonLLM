import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.db import engine
from app.models.snack import Snack  # 스낵 모델 import


async def get_snack_data():
    async with AsyncSession(engine) as session:
        # 첫 번째 스낵 데이터 조회
        query = select(Snack).limit(1)
        result = await session.execute(query)
        snack = result.scalar_one_or_none()

        if snack:
            print(f"스낵 이름: {snack.name}")
            print(f"스낵 설명: {snack.description}")
        else:
            print("데이터가 없습니다.")


# 테스트 실행
if __name__ == "__main__":
    asyncio.run(get_snack_data())
