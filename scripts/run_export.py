import asyncio
from app.api.deps import get_db
from scripts.export_data_to_json import (
    export_cmetadata_to_json,
)  # 실제 함수가 있는 파일 경로에 맞게 수정
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main():
    async for session in get_db():
        await export_cmetadata_to_json(session)
        break  # get_db는 generator이므로 한 번만 사용


if __name__ == "__main__":
    asyncio.run(main())
