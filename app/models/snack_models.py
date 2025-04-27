from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String, BigInteger, Integer, Boolean, Double, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB

from .snack_types import SnackNutrient, ServingSize, SnackAdditiveGrade, SafeFoodMark

table_schema = "public"


class MapSnackItemAdditive(SQLModel, table=True):
    """
    간식 아이템 - 첨가물 매핑 정보
    """
    __tablename__ = "map_snack_item_additive"
    __table_args__ = {"schema": table_schema}

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column())
    modified_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    modified_at: Optional[datetime] = Field(default=None, sa_column=Column())
    snack_item_id: int = Field(foreign_key=f"{table_schema}.snack_item.id")
    snack_additive_id: int = Field(foreign_key=f"{table_schema}.snack_additive.id")


class Snack(SQLModel, table=True):
    """
    간식 정보
    """
    __tablename__ = "snack"
    __table_args__ = {"schema": table_schema}

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column())
    modified_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    modified_at: Optional[datetime] = Field(default=None, sa_column=Column())
    barcode: Optional[str] = Field(default=None, sa_column=Column(String(255), unique=True))
    snack_type: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    name: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    company: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    total_serving_size: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    thumbnail_url: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    main_image_url: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    allergy_list: Optional[List[str]] = Field(default=None, sa_column=Column(JSONB))
    safe_food_mark_list: Optional[List[SafeFoodMark]] = Field(default=None, sa_column=Column(JSONB))
    total_serving_size_num: Optional[float] = Field(default=None, sa_column=Column(Double))

    snack_item: Optional["SnackItem"] = Relationship(back_populates="snack")


class SnackItem(SQLModel, table=True):
    """
    간식 아이템 정보
    """
    __tablename__ = "snack_item"
    __table_args__ = {"schema": table_schema}

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column())
    modified_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    modified_at: Optional[datetime] = Field(default=None, sa_column=Column())
    snack_id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, ForeignKey(f"{table_schema}.snack.id")))
    item_name: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    service_unit: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    nutrient_list: Optional[List[SnackNutrient]] = Field(default=None, sa_column=Column(JSONB))
    calorie: Optional[int] = Field(default=None, sa_column=Column(Integer))
    serving_size: Optional[ServingSize] = Field(default=None, sa_column=Column(JSONB))

    snack: Optional[Snack] = Relationship(back_populates="snack_item")
    additives: List["SnackAdditive"] = Relationship(
        back_populates="snack_items",
        link_model=MapSnackItemAdditive
    )


class SnackAdditive(SQLModel, table=True):
    """
    간식 첨가물 정보
    """
    __tablename__ = "snack_additive"
    __table_args__ = {"schema": table_schema}

    id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, primary_key=True))
    created_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    created_at: Optional[datetime] = Field(default=None, sa_column=Column())
    modified_by: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    modified_at: Optional[datetime] = Field(default=None, sa_column=Column())
    korean_name: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    english_name: Optional[str] = Field(default=None, sa_column=Column(String(255)))
    main_use_list: Optional[List[str]] = Field(default=None, sa_column=Column(JSONB))
    grade: Optional[SnackAdditiveGrade] = Field(default=None, sa_column=Column(String(255)))
    description: Optional[str] = Field(default=None, sa_column=Column(String(2048)))
    stability_message: Optional[str] = Field(default=None, sa_column=Column(String(2048)))
    simple_name_list: Optional[List[str]] = Field(default=None, sa_column=Column(JSONB))
    is_main_use_type: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))

    snack_items: List["SnackItem"] = Relationship(
        back_populates="additives",
        link_model=MapSnackItemAdditive
    )
