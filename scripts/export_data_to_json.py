import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import text
import json
from pathlib import Path

from app.api.deps import get_db


async def export_cmetadata_to_json(
    db_session: AsyncSession, output_path: str = "scripts/sample_data.json"
):
    # cmetadata는 JSON 문자열 형식이므로, 파싱 필요
    query = text("SELECT cmetadata FROM langchain_pg_embedding LIMIT 30")
    result = await db_session.execute(query)

    rows = result.fetchall()
    cmetadata_list = [row[0] for row in rows]  # row is a tuple like (cmetadata,)

    # 저장할 디렉토리 확인 및 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cmetadata_list, f, ensure_ascii=False, indent=2)
