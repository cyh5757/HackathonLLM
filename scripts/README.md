# 스크립트 파일 설명

## 데이터 변환 및 생성
- `export_data_to_json.py`: PostgreSQL 데이터베이스에 저장된 임베딩 데이터를 JSON 형식으로 변환하는 스크립트
- `run_export.py`: `export_data_to_json.py`를 실행하여 JSON 파일을 생성하는 실행 스크립트

## 샘플 데이터 생성
- `sample_json2csv.py`: JSON 형식의 샘플 데이터를 CSV로 변환하고 규칙 기반 질문-답변 쌍을 생성
- `allergy_qa_sample.csv`: 생성된 샘플 질문-답변 데이터셋

## RAG 시스템 평가
- `evaluate_rag.py`: 일반 RAG(baseline)와 리랭킹 RAG의 응답을 비교 평가
- `rag_response_comparison.csv`: 두 RAG 시스템의 응답 결과를 저장한 파일
- `openai_eval.py`: GPT-4를 사용하여 두 RAG 시스템의 응답 품질을 비교 평가
- `rag_gpt_comparison.csv`: GPT-4 평가 결과를 저장한 파일

1차 결과 : baseline : 16, rerank : 4
예상 이유 : numbering을 통해 rerank를 했지만, 그 과정에서 질 좋은 평가라 생각하지 않음. 왜냐하면 그 기준이 모호하고, numbering을 어떻게 표현해야하는지 인지하지 못함.
이후 행동 : numbering에 reasoning을 추가해서, 좀더 다채로운 평가를 보고 rerank하는 걸로 변경
---------------------------------
2차 결과 : baseline 15 , rerank 5
reasoning을 이용한 성능 개선.

결과: rerank를 했을시에 좀더 예상하는 성능 개선이 있을거라 예측했지만 아쉽게도 그렇지 않았다.
이유 예상 : rule base 기반 질문과 정답지로 이뤄져있어서, 평가하기에 더 많은 