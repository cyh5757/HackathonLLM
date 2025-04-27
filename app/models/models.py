from typing import List, Optional

from app.models import BaseEntity
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlmodel import Field

table_schema = "public"


# PGVector를 사용하지않고 직접 sqlalchemy 으로 DB에 접근할 때 정의해서 사용.
class LangchainPgEmbedding(BaseEntity, table=True):
    __tablename__ = "langchain_pg_embedding"
    __table_args__ = {"schema": table_schema}

    id: str = Field(
        sa_column=Column(Text, primary_key=True, nullable=False)
    )

    collection_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            UUID,
            ForeignKey(f"{table_schema}.langchain_pg_collection.id", ondelete="CASCADE"),
            nullable=True
        )
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536))
    )

    document: Optional[str] = Field(
        default=None,
        sa_column=Column(Text)
    )

    cmetadata: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSONB)
    )
