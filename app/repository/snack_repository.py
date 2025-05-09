from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlmodel import select

from app.models.snack_dto import SnackDetailDto
from app.models.snack_models import Snack, SnackItem
from app.models.snack_types import AllergyKeyword


# async def get_snack_detail_by_id(db_session: AsyncSession, snack_id: int) -> Optional[SnackDetailDto]:
#     """
#     snack_id로 Snack + SnackItem + SnackAdditive까지 한방 조회하여 DTO 반환
#     """
#     stmt = (
#         select(Snack)
#         .where(Snack.id == snack_id)
#         .options(
#             joinedload(Snack.snack_item)
#             .joinedload(SnackItem.additives)
#         )
#     )

#     result = await db_session.exec(stmt)
#     snack = result.one_or_none()
#     return SnackDetailDto.from_model(snack) if snack else None


async def get_snack_detail_by_id(
    db_session: AsyncSession, snack_id: int
) -> Optional[SnackDetailDto]:
    """
    snack_id로 Snack + SnackItem + SnackAdditive까지 한방 조회하여 DTO 반환
    """
    stmt = (
        select(Snack)
        .where(Snack.id == snack_id)
        .options(joinedload(Snack.snack_item).joinedload(SnackItem.additives))
    )

    result = await db_session.exec(stmt)
    snack = result.unique().one_or_none()
    return SnackDetailDto.from_model(snack) if snack else None


async def get_snacks_by_allergy_keyword(
    db_session: AsyncSession, keyword: AllergyKeyword, limit: int = 10
) -> List[SnackDetailDto]:
    stmt = (
        select(Snack)
        .where(Snack.allergy_list.contains([keyword.value]))
        .options(joinedload(Snack.snack_item).joinedload(SnackItem.additives))
        .limit(limit)
    )
    result = await db_session.exec(stmt)
    snacks = result.unique().all()
    return [SnackDetailDto.from_model(snack) for snack in snacks]


async def get_snacks_without_allergy_keyword(
    db_session: AsyncSession, keyword: AllergyKeyword, limit: int = 10
) -> List[SnackDetailDto]:
    """
    알러지 JSON 필드에 특정 알러지 키워드가 **포함되지 않은** snack을 조회
    """
    stmt = (
        select(Snack)
        .where(~Snack.allergy_list.contains([keyword.value]))
        .options(joinedload(Snack.snack_item).joinedload(SnackItem.additives))
        .limit(limit)
    )
    result = await db_session.exec(stmt)
    snacks = result.unique().all()
    return [SnackDetailDto.from_model(snack) for snack in snacks]
