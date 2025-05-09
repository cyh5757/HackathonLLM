import requests

res = requests.get("http://localhost:8000/api/v1/snacks/test")
print("Status code:", res.status_code)
print("Response text:", res.text)

try:
    print("JSON:", res.json())
except requests.exceptions.JSONDecodeError:
    print("⚠️ 응답이 JSON 형식이 아닙니다.")
