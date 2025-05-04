import json
import random
import pandas as pd

# Load the sample data JSON
with open("sample_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)


# Define a function to create a natural language question about allergen risk
def generate_allergy_question(snack):
    additives = ", ".join(snack.get("additives", []))
    return f"'{snack['name']}' 제품에 포함된 첨가물({additives}) 중 알레르기 유발 가능성이 있는 성분이 있는지 알려주세요."


# Select 20 random items and generate Q&A pairs
samples = random.sample(data, 20)
qa_pairs = []

for item in samples:
    question = generate_allergy_question(item)
    answer = f"{item['name']} 제품은 총 {len(item['additives'])}개의 첨가물이 포함되어 있으며, 안전등급은 다음과 같습니다: {', '.join(item['additive_grades'])}."
    qa_pairs.append({"제품명": item["name"], "질문": question, "모델대답예시": answer})

# Convert to DataFrame for output
qa_df = pd.DataFrame(qa_pairs)

# Save to CSV file
csv_path = "allergy_qa_sample.csv"
qa_df.to_csv(csv_path, index=False)
