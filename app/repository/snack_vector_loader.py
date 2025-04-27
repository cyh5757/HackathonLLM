import logging
import json
import os
from typing import List, Dict, Any

from app.api.deps import SessionDep
from app.core import agent_tools
from app.repository import pgvector_repository
from langchain_core.documents import Document
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def load_db_data_to_vectordb(
    session: SessionDep,
    batch_size: int = 100,
    query_batch_size: int = 500,  # 쿼리당 데이터 조회 개수
    limit: int = 1000000  # 실질적으로 제한 없음
):
    """
    DB에 이미 저장된 과자 데이터와 첨가물 데이터를 벡터 데이터베이스에 로드합니다.
    
    Args:
        session: 데이터베이스 세션
        batch_size: 한 번에 처리할 문서 수
        query_batch_size: 쿼리당 조회할 데이터 개수
        limit: 총 임베딩할 데이터 개수 (기본값 매우 큼 = 제한 없음)
    """
    logger.info(f"DB에서 과자 및 첨가물 데이터 로드 시작")
    
    # 다양한 데이터 타입 처리를 위한 문서 리스트
    documents = []
    
    # 과자 데이터 배치 처리
    offset = 0
    snack_count = 0
    
    while snack_count < limit // 2:
        # DB에서 과자 데이터 배치 조회
        snack_query = text(f"""
            WITH SnackAdditives AS (
                SELECT 
                    s.id as snack_id,
                    array_agg(sa.korean_name) as additives,
                    array_agg(sa.description) as additive_descriptions,
                    array_agg(sa.grade) as additive_grades,
                    array_agg(sa.main_use_list) as additive_main_uses
                FROM 
                    snack s
                LEFT JOIN 
                    snack_item si ON s.id = si.snack_id
                LEFT JOIN 
                    map_snack_item_additive msia ON si.id = msia.snack_item_id
                LEFT JOIN 
                    snack_additive sa ON msia.snack_additive_id = sa.id
                GROUP BY 
                    s.id
            ),
            SimilarProducts AS (
                SELECT 
                    s.id as snack_id,
                    array_agg(DISTINCT s2.id) FILTER (WHERE s2.company = s.company AND s2.id != s.id) as same_company_products,
                    array_agg(DISTINCT s2.id) FILTER (WHERE s2.snack_type = s.snack_type AND s2.id != s.id) as same_type_products
                FROM 
                    snack s
                JOIN 
                    snack s2 ON (s2.company = s.company OR s2.snack_type = s.snack_type) AND s2.id != s.id
                GROUP BY 
                    s.id
            ),
            NutritionInfo AS (
                SELECT 
                    s.id as snack_id,
                    si.nutrient_list,
                    si.calorie,
                    si.serving_size
                FROM 
                    snack s
                LEFT JOIN 
                    snack_item si ON s.id = si.snack_id
                WHERE 
                    si.id IS NOT NULL
                LIMIT 1
            )
            SELECT 
                s.id, s.barcode, s.snack_type, s.name, s.company,
                s.total_serving_size, s.allergy_list, s.safe_food_mark_list,
                s.thumbnail_url, s.main_image_url,
                COALESCE(sa.additives, '{{}}') as additives,
                COALESCE(sa.additive_descriptions, '{{}}') as additive_descriptions,
                COALESCE(sa.additive_grades, '{{}}') as additive_grades,
                COALESCE(sa.additive_main_uses, '{{}}') as additive_main_uses,
                COALESCE(sp.same_company_products, '{{}}') as same_company_products,
                COALESCE(sp.same_type_products, '{{}}') as same_type_products,
                ni.nutrient_list, ni.calorie, ni.serving_size
            FROM 
                snack s
            LEFT JOIN 
                SnackAdditives sa ON s.id = sa.snack_id
            LEFT JOIN 
                SimilarProducts sp ON s.id = sp.snack_id
            LEFT JOIN 
                NutritionInfo ni ON s.id = ni.snack_id
            ORDER BY 
                s.id
            LIMIT {query_batch_size} OFFSET {offset}
        """)
        
        snack_result = await session.execute(snack_query)
        snack_data_batch = snack_result.mappings().all()
        
        if not snack_data_batch:
            break  # 더 이상 데이터가 없으면 루프 종료
            
        logger.info(f"DB에서 과자 데이터 배치 {offset}-{offset+len(snack_data_batch)} 로드 완료")
        
        # 과자 데이터 문서화 (기존 로직 유지)
        for snack in snack_data_batch:
            # JSON 형태로 저장된 리스트 필드 파싱
            allergy_list = json.loads(snack.get('allergy_list', '[]')) if isinstance(snack.get('allergy_list'), str) else snack.get('allergy_list', [])
            safe_food_mark_list = json.loads(snack.get('safe_food_mark_list', '[]')) if isinstance(snack.get('safe_food_mark_list'), str) else snack.get('safe_food_mark_list', [])
            
            # 첨가물 정보 파싱
            additives = snack.get('additives', [])
            if isinstance(additives, str):
                try:
                    additives = json.loads(additives.replace('{{', '[').replace('}}', ']'))
                except:
                    additives = []
            
            additive_descriptions = snack.get('additive_descriptions', [])
            if isinstance(additive_descriptions, str):
                try:
                    additive_descriptions = json.loads(additive_descriptions.replace('{{', '[').replace('}}', ']'))
                except:
                    additive_descriptions = []
            
            # 첨가물 등급 파싱
            additive_grades = snack.get('additive_grades', [])
            if isinstance(additive_grades, str):
                try:
                    additive_grades = json.loads(additive_grades.replace('{{', '[').replace('}}', ']'))
                except:
                    additive_grades = []
            
            # 첨가물 주요 용도 파싱
            additive_main_uses = snack.get('additive_main_uses', [])
            if isinstance(additive_main_uses, str):
                try:
                    additive_main_uses = json.loads(additive_main_uses.replace('{{', '[').replace('}}', ']'))
                except:
                    additive_main_uses = []
            
            # 영양 정보 파싱
            nutrient_list = snack.get('nutrient_list', None)
            if isinstance(nutrient_list, str):
                try:
                    nutrient_list = json.loads(nutrient_list)
                except:
                    nutrient_list = {}
            
            # 같은 회사의 다른 제품, 같은 타입의 다른 제품 파싱
            same_company_products = snack.get('same_company_products', [])
            if isinstance(same_company_products, str):
                try:
                    same_company_products = json.loads(same_company_products.replace('{{', '[').replace('}}', ']'))
                except:
                    same_company_products = []
                    
            same_type_products = snack.get('same_type_products', [])
            if isinstance(same_type_products, str):
                try:
                    same_type_products = json.loads(same_type_products.replace('{{', '[').replace('}}', ']'))
                except:
                    same_type_products = []
            
            # 첨가물 안전성 요약 계산
            safety_counts = {"SAFE": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "UNKNOWN": 0}
            for grade in additive_grades:
                if grade:
                    safety_counts[grade if grade in safety_counts else "UNKNOWN"] += 1
            
            safety_summary = ""
            if safety_counts["SAFE"] > 0:
                safety_summary += f"안전 첨가물 {safety_counts['SAFE']}개, "
            if safety_counts["LOW"] > 0:
                safety_summary += f"저위험 첨가물 {safety_counts['LOW']}개, "
            if safety_counts["MODERATE"] > 0:
                safety_summary += f"중간위험 첨가물 {safety_counts['MODERATE']}개, "
            if safety_counts["HIGH"] > 0:
                safety_summary += f"고위험 첨가물 {safety_counts['HIGH']}개, "
            if safety_counts["UNKNOWN"] > 0:
                safety_summary += f"위험도 불명 첨가물 {safety_counts['UNKNOWN']}개, "
                
            safety_summary = safety_summary.rstrip(", ") if safety_summary else "첨가물 안전성 정보 없음"
            
            # 첨가물 주요 용도 요약
            use_counts = {}
            for uses in additive_main_uses:
                if uses:
                    uses_list = json.loads(uses) if isinstance(uses, str) else uses
                    for use in uses_list:
                        use_counts[use] = use_counts.get(use, 0) + 1
            
            top_uses = sorted(use_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            use_summary = ", ".join([f"{use}({count})" for use, count in top_uses]) if top_uses else "첨가물 용도 정보 없음"
            
            # 첨가물 정보 텍스트 구성
            additives_info = ""
            for i, additive in enumerate(additives):
                if i < len(additive_descriptions):
                    description = additive_descriptions[i]
                    grade = additive_grades[i] if i < len(additive_grades) else "정보 없음"
                    grade_text = f"({grade})" if grade else ""
                    additives_info += f"\n- {additive}{grade_text}: {description}"
                else:
                    additives_info += f"\n- {additive}"
            
            # 영양 정보 텍스트 구성
            calorie = snack.get('calorie', None)
            serving_size = snack.get('serving_size', None)
            
            nutrition_info = ""
            if calorie is not None:
                nutrition_info += f"\n칼로리: {calorie}kcal"
            if serving_size is not None:
                nutrition_info += f"\n1회 제공량: {serving_size}"
            
            if nutrient_list:
                nutrition_info += "\n영양성분:"
                # nutrient_list가 딕셔너리인 경우
                if isinstance(nutrient_list, dict):
                    for nutrient, value in nutrient_list.items():
                        nutrition_info += f"\n- {nutrient}: {value}"
                # nutrient_list가 리스트인 경우
                elif isinstance(nutrient_list, list):
                    for item in nutrient_list:
                        if isinstance(item, dict) and 'name' in item and 'value' in item:
                            nutrition_info += f"\n- {item['name']}: {item['value']}"
                        else:
                            nutrition_info += f"\n- {item}"
            
            if not nutrition_info:
                nutrition_info = "\n영양 정보 없음"
            
            # 과자 기본 정보 - 확장된 정보 포함
            snack_doc = Document(
                page_content=f"""
과자 정보:
이름: {snack.get('name', '정보 없음')}
바코드: {snack.get('barcode', '정보 없음')}
종류: {snack.get('snack_type', '정보 없음')}
제조사: {snack.get('company', '정보 없음')}
총 제공량: {snack.get('total_serving_size', '정보 없음')}
알레르기 정보: {', '.join(allergy_list) if allergy_list else '정보 없음'}
안전 식품 마크: {', '.join(safe_food_mark_list) if safe_food_mark_list else '정보 없음'}

영양 정보:{nutrition_info}

첨가물 안전성 요약: {safety_summary}
첨가물 주요 용도: {use_summary}

포함된 첨가물:{additives_info if additives_info else ' 첨가물 정보 없음'}
                """.strip(),
                metadata={
                    "id": snack.get('id'),
                    "type": "snack",
                    "name": snack.get('name', ''),
                    "barcode": snack.get('barcode', ''),
                    "snack_type": snack.get('snack_type', ''),
                    "company": snack.get('company', ''),
                    "thumbnail_url": snack.get('thumbnail_url', ''),
                    "main_image_url": snack.get('main_image_url', ''),
                    "additives": additives,
                    "additive_descriptions": additive_descriptions,
                    "additive_grades": additive_grades,
                    "calorie": calorie,
                    "safety_counts": safety_counts,
                    "top_additive_uses": [use for use, _ in top_uses] if top_uses else [],
                    "similar_company_products": same_company_products[:5] if same_company_products else [],
                    "similar_type_products": same_type_products[:5] if same_type_products else [],
                    "prefix": "snack",
                    "field_name": "snack_data"
                }
            )
            documents.append(snack_doc)
        
        # 다음 배치로 이동
        offset += len(snack_data_batch)
        snack_count += len(snack_data_batch)
        
        # 데이터를 임베딩하고 저장 (메모리 효율성을 위해)
        if len(documents) >= batch_size:
            await save_documents_batch(session, documents, batch_size)
            documents = []  # 메모리 정리
        
        # 더 이상 데이터가 없으면 종료
        if len(snack_data_batch) < query_batch_size:
            break
    
    # 첨가물 데이터 배치 처리
    offset = 0
    additive_count = 0
    
    while additive_count < limit // 2:
        # DB에서 첨가물 데이터 배치 조회
        additive_query = text(f"""
            SELECT id, korean_name, english_name, main_use_list, 
                   grade, description, stability_message
            FROM snack_additive
            ORDER BY id
            LIMIT {query_batch_size} OFFSET {offset}
        """)
        additive_result = await session.execute(additive_query)
        additive_data_batch = additive_result.mappings().all()
        
        if not additive_data_batch:
            break  # 더 이상 데이터가 없으면 루프 종료
            
        logger.info(f"DB에서 첨가물 데이터 배치 {offset}-{offset+len(additive_data_batch)} 로드 완료")
        
        # 첨가물 데이터 문서화
        for additive in additive_data_batch:
            # 주요 용도 리스트 파싱
            main_use_list = json.loads(additive.get('main_use_list', '[]')) if isinstance(additive.get('main_use_list'), str) else additive.get('main_use_list', [])
            
            # 첨가물 정보 - 지정된 필드만 포함
            additive_doc = Document(
                page_content=f"""
첨가물 정보:
한글명: {additive.get('korean_name', '정보 없음')}
영문명: {additive.get('english_name', '정보 없음')}
주요 용도: {', '.join(main_use_list) if main_use_list else '정보 없음'}
등급: {additive.get('grade', '정보 없음')}
설명: {additive.get('description', '정보 없음')}
안전성 메시지: {additive.get('stability_message', '정보 없음')}
                """.strip(),
                metadata={
                    "id": additive.get('id'),
                    "type": "additive",
                    "korean_name": additive.get('korean_name', ''),
                    "english_name": additive.get('english_name', ''),
                    "grade": additive.get('grade', ''),
                    "main_use_list": main_use_list,
                    "prefix": "snack",
                    "field_name": "snack_data"
                }
            )
            documents.append(additive_doc)
        
        # 다음 배치로 이동
        offset += len(additive_data_batch)
        additive_count += len(additive_data_batch)
        
        # 데이터를 임베딩하고 저장 (메모리 효율성을 위해)
        if len(documents) >= batch_size:
            await save_documents_batch(session, documents, batch_size)
            documents = []  # 메모리 정리
            
        # 더 이상 데이터가 없으면 종료
        if len(additive_data_batch) < query_batch_size:
            break
    
    # 남은 문서 처리
    if documents:
        await save_documents_batch(session, documents, batch_size)
    
    logger.info(f"DB에서 과자 및 첨가물 데이터 로드 완료 (과자: {snack_count}, 첨가물: {additive_count})")
    return snack_count + additive_count

async def save_documents_batch(session, documents, batch_size):
    """배치 단위로 문서를 벡터 DB에 저장합니다."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        await insert_documents_to_vectordb(session, batch)
        logger.info(f"문서 배치 {i}-{i+len(batch)}/{len(documents)} 벡터 DB에 저장 완료")

async def insert_documents_to_vectordb(
    session: SessionDep,
    documents: List[Document]
):
    """문서 리스트를 벡터 DB에 저장합니다."""
    for doc in documents:
        await pgvector_repository.insert_memory(
            db_session=session,
            new_memory=doc,
            key_prefix=doc.metadata.get("prefix", "snack"),
            field_name=doc.metadata.get("field_name", "snack_data")
        )
    await session.commit()

# 데이터 로드 스크립트를 직접 실행할 때 사용
if __name__ == "__main__":
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession

    
    async def main():
        # 여기서는 세션을 가정
        # 실제 구현에서는 적절한 세션 관리 필요
        session = AsyncSession()
        try:
            count = await load_db_data_to_vectordb(session)
            logger.info(f"총 {count}개의 문서가 벡터 DB에 저장되었습니다.")
        finally:
            await session.close()
    

    asyncio.run(main()) 