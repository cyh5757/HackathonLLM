import asyncio
import sys

# 이거 추가! (Windows 비동기 event loop 호환성 맞춰줌)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import asyncio
import logging

from app.api.deps import get_db
from app.services.load_snack_service import load_db_data_to_vectordb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    async for session in get_db():
        await load_db_data_to_vectordb(session=session, batch_size=100)


if __name__ == "__main__":
    asyncio.run(main())


#set PYTHONPATH=.