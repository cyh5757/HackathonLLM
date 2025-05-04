import pandas as pd
from openai import OpenAI
import time

from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# ✅ CSV 불러오기
df = pd.read_csv("scripts/rag_response_comparison_2.csv")


# ✅ 평가 함수 정의
def get_gpt_vote(question, example, baseline, rerank):
    prompt = f"""
질문: {question}
예시 응답: {example}
응답1 (baseline): {baseline}
응답2 (rerank): {rerank}

당신은 품질 평가자입니다.
예시 응답을 참고해, 응답1과 응답2 중 질문에 더 정확하고 충실하게 답한 것을 1 또는 2로 숫자만 선택해주세요.
다른 설명 없이 숫자 하나만 출력하세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "너는 사용자의 질문에 대한 두 답변 중 어떤 것이 더 나은지 판단하는 평가자야.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return "error"


# ✅ GPT 평가 실행
results = []
for i, row in df.iterrows():
    vote = get_gpt_vote(
        row["질문"], row["모델대답예시"], row["baseline_응답"], row["rerank_응답"]
    )
    results.append(vote)
    time.sleep(1.5)  # OpenAI API 요금/속도 제한 방지

# ✅ 결과 저장
df["GPT_평가"] = results
df.to_csv("scripts/rag_gpt_comparison_2.csv", index=False, encoding="utf-8-sig")
print("✅ 평가 완료 및 저장되었습니다: scripts/rag_gpt_comparison_2.csv")
