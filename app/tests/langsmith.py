import os

import dotenv

dotenv.load_dotenv()


def set_langsmith(project_name=None, set_enable=True):
    if set_enable:
        langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
        langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")
        # 더 긴 API 키 선택
        result = langchain_key if len(langchain_key.strip()) >= len(langsmith_key.strip()) else langsmith_key
        if result.strip() == "":
            print("\nLangChain/LangSmith API Key가 설정되지 않았습니다.")
            return
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project_name
        print(f"\nLangSmith 추적을 시작합니다.\n[프로젝트명]\n{project_name}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        print("\nLangSmith 추적을 하지 않습니다.")
