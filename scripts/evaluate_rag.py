import asyncio
import pandas as pd
from app.services.rag_service_test import (
    generate_rag_response,
    generate_rag_response_rerank,
)  # 경로는 사용자 환경에 따라 조정
from app.api.deps import SessionDep  # FastAPI 종속성
from contextlib import asynccontextmanager
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 평가할 질문 목록 로드
qa_df = pd.read_csv("scripts/allergy_qa_sample.csv")

results = []


@asynccontextmanager
async def get_session():
    # 실제 환경에서는 db 세션 또는 더미 세션 생성 로직 필요
    yield SessionDep()  # 대체 필요 시 여기에 세션 생성 로직 추가


async def run_evaluation():
    async with get_session() as session:
        for idx, row in qa_df.iterrows():
            query = row["질문"]
            product = row["제품명"]

            try:
                baseline_response = await generate_rag_response(session, query)
            except Exception as e:
                baseline_response = f"❌ 오류: {str(e)}"

            try:
                rerank_response = await generate_rag_response_rerank(session, query)
            except Exception as e:
                rerank_response = f"❌ 오류: {str(e)}"

            results.append(
                {
                    "제품명": product,
                    "질문": query,
                    "baseline_응답": baseline_response,
                    "rerank_응답": rerank_response,
                    "모델대답예시": row["모델대답예시"],
                }
            )

    # 결과 저장
    results_df = pd.DataFrame(results)
    output_path = "scripts/rag_response_comparison_2.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 결과 저장 완료: {output_path}")


# 실행
asyncio.run(run_evaluation())
