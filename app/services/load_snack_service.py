import logging
import json
from typing import List

from app.api.deps import SessionDep
from app.repository import pgvector_repository
from sqlalchemy import text
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_db_data_to_vectordb(
    session: SessionDep,
    batch_size: int = 100,
):
    """
    snack 테이블의 데이터를 벡터DB에 임베딩하여 저장하는 함수.
    
    Args:
        session: 데이터베이스 세션
        batch_size: 한 번에 몇 개 문서를 벡터DB에 넣을지 설정
    """
    logger.info(f"DB에서 snack 데이터를 가져와 벡터DB에 저장합니다.")

    offset = 0
    documents = []

    while True:
        query = text(f"""
            SELECT id, name, barcode, snack_type, company, thumbnail_url, main_image_url
            FROM snack
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """)

        result = await session.execute(query)
        rows = result.mappings().all()

        if not rows:
            break

        logger.info(f"{len(rows)}개의 snack 데이터를 불러왔습니다. ({offset}부터 시작)")

        for row in rows:
            page_content = f"""
이름: {row['name']}
바코드: {row['barcode']}
종류: {row['snack_type']}
제조사: {row['company']}
            """.strip()

            document = Document(
                page_content=page_content,
                metadata={
                    "id": row["id"],
                    "name": row["name"],
                    "barcode": row["barcode"],
                    "snack_type": row["snack_type"],
                    "company": row["company"],
                    "thumbnail_url": row["thumbnail_url"],
                    "main_image_url": row["main_image_url"],
                    "prefix": "snack",
                    "field_name": "snack_data",
                }
            )

            documents.append(document)

        offset += batch_size

        # 벡터DB에 저장
        await save_documents_to_vectordb(session, documents)
        documents = []  # 메모리 비우기

    logger.info("DB snack 데이터 → 벡터DB 저장 완료 ✅")


async def save_documents_to_vectordb(
    session: SessionDep,
    documents: List[Document]
):
    """주어진 문서 리스트를 벡터DB에 저장합니다."""
    for doc in documents:
        await pgvector_repository.insert_memory(
            db_session=session,
            new_memory=doc,
            key_prefix=doc.metadata["prefix"],
            field_name=doc.metadata["field_name"]
        )
    await session.commit()
