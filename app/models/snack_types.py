from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ServingSize(BaseModel):
    g: Optional[float] = None
    ml: Optional[float] = None


class SnackNutrient(BaseModel):
    name: str
    unit: str
    amount: float
    daily_percent: Optional[float] = None


class SnackInfo(BaseModel):
    allergy_list: List[str]
    safe_food_mark_list: List[str]


class AdditiveInfo(BaseModel):
    main_use_list: List[str]
    simple_name_list: List[str]


class SnackAdditiveGrade(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"

    @property
    def message(self) -> str:
        return {
            self.SAFE: "비교적 안전하다고 알려진 식품 첨가물이에요.",
            self.LOW: "인체에 영향은 크지 않지만 다량 섭취 시 주의가 필요한 식품 첨가물이에요.",
            self.MODERATE: "인체에 영향은 크지 않지만 다량 섭취 시 주의가 필요한 식품 첨가물이에요.",
            self.HIGH: "건강에 부정적인 영향을 줄 수 있는 식품 첨가물이에요. 가능한 한 섭취를 피하거나, 최소화하는 것이 좋겠어요.",
            self.UNKNOWN: "이 식품 첨가물에 대한 충분한 연구 데이터가 부족하거나 안전성이 확인되지 않은 식품 첨가물이에요.",
        }[self]

    @property
    def keyword(self) -> str:
        return {
            self.SAFE: "안전 (No)",
            self.LOW: "저위험 (Low)",
            self.MODERATE: "중위험 (Moderate)",
            self.HIGH: "고위험 (High)",
            self.UNKNOWN: "분석불가 (Unknown)",
        }[self]


class SafeFoodMark(str, Enum):
    HACCP = "HACCP"
    KIDS_SAFE = "KIDS_SAFE"
    HFF = "HFF"
    GMP = "GMP"
    TRACEABILITY_FOOD = "TRACEABILITY_FOOD"

    @property
    def message(self) -> str:
        return {
            self.HACCP: "원료 관리부터 유통까지 모든 과정에서 기준을 충족한 위생적이고 안전한 시설에서 만들어진 식품이에요.",
            self.KIDS_SAFE: "어린이 기호식품 중 품질을 인정받은 안전하고 영양을 고루 갖춘 식품이에요.",
            self.HFF: "인체에 유용한 기능성을 가진 원료로 제조되어 기능성과 안전성이 인정된 식품이에요.",
            self.GMP: "체계적인 관리와 엄격한 제조 기준을 통해 안전하고 우수한 품질로 만들어진 건강식품이에요.",
            self.TRACEABILITY_FOOD: "제조, 가공, 판매까지 모든 단계에서 이력 추적 정보를 기록하고 관리하여, 유통 과정을 확인할 수 있는 식품이에요.",
        }[self]


class AllergyKeyword(str, Enum):
    가리비 = "가리비"
    고등어 = "고등어"
    굴 = "굴"
    글루텐 = "글루텐"
    견과류 = "견과류"
    게 = "게"
    계란 = "계란"
    난류 = "난류"
    닭가슴살 = "닭가슴살"
    닭고기 = "닭고기"
    돼지고기 = "돼지고기"
    돼지_돈피 = "돼지(돈피)"
    대두 = "대두"
    대두유 = "대두유"
    땅콩 = "땅콩"
    마가린 = "마가린"
    마카다미아 = "마카다미아"
    메밀 = "메밀"
    메타중아황산나트륨 = "메타중아황산나트륨"
    모시조개 = "모시조개"
    바지락 = "바지락"
    복숭아 = "복숭아"
    보리 = "보리"
    새우 = "새우"
    새우젓 = "새우젓"
    쇠고기 = "쇠고기"
    아몬드 = "아몬드"
    알류 = "알류"
    아황산나트륨 = "아황산나트륨"
    아황산류 = "아황산류"
    옥수수 = "옥수수"
    오징어 = "오징어"
    우유 = "우유"
    잣 = "잣"
    전복 = "전복"
    조개류 = "조개류"
    천도복숭아 = "천도복숭아"
    참깨 = "참깨"
    페닐알라닌 = "페닐알라닌"
    피칸 = "피칸"
    피스타치오 = "피스타치오"
    홍합 = "홍합"
    젤라틴_돼지 = "젤라틴(돼지)"
    젤라틴_쇠고기 = "젤라틴(쇠고기)"
    헤이즐넛 = "헤이즐넛"
    토마토 = "토마토"
    너트 = "너트"
