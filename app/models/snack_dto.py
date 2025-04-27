from typing import List, Optional
from pydantic import BaseModel
from app.models.snack_models import Snack, SnackItem, SnackAdditive


class SnackAdditiveDto(BaseModel):
    id: int
    korean_name: Optional[str]
    english_name: Optional[str]
    grade: Optional[str]
    description: Optional[str]
    is_main_use_type: bool


class SnackItemDto(BaseModel):
    id: int
    item_name: Optional[str]
    service_unit: Optional[str]
    calorie: Optional[int]
    nutrient_list: List[dict]
    serving_size: dict
    additives: List[SnackAdditiveDto]


class SnackDetailDto(BaseModel):
    id: int
    name: Optional[str]
    barcode: Optional[str]
    company: Optional[str]
    snack_type: Optional[str]
    total_serving_size: Optional[str]
    thumbnail_url: Optional[str]
    main_image_url: Optional[str]
    allergy_list: List[str]
    safe_food_mark_list: List[str]
    total_serving_size_num: Optional[float]
    item: SnackItemDto

    @classmethod
    def from_model(cls, snack: Snack) -> "SnackDetailDto":
        item = snack.snack_item

        additives = [
            SnackAdditiveDto(
                id=add.id,
                korean_name=add.korean_name,
                english_name=add.english_name,
                grade=add.grade,
                description=add.description,
                is_main_use_type=add.is_main_use_type,
            )
            for add in (item.additives or [])
        ]

        serving_size = {}
        if item and item.serving_size:
            if isinstance(item.serving_size, dict):
                serving_size = item.serving_size
            elif hasattr(item.serving_size, "dict"):
                serving_size = item.serving_size.dict()

        item_dto = SnackItemDto(
            id=item.id,
            item_name=item.item_name,
            service_unit=item.service_unit,
            calorie=item.calorie,
            nutrient_list=item.nutrient_list or [],
            serving_size=serving_size,
            additives=additives,
        )

        return cls(
            id=snack.id,
            name=snack.name,
            barcode=snack.barcode,
            company=snack.company,
            snack_type=snack.snack_type,
            total_serving_size=snack.total_serving_size,
            thumbnail_url=snack.thumbnail_url,
            main_image_url=snack.main_image_url,
            allergy_list=snack.allergy_list or [],
            safe_food_mark_list=[str(m) for m in snack.safe_food_mark_list or []],
            total_serving_size_num=snack.total_serving_size_num,
            item=item_dto,
        )
