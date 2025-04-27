import logging

from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import DistanceStrategy


class GenerizedMetadataByID:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    def generate_metadata(self, context: str) -> str:
        prompt = f"다음 문서의 메타데이터의 ID로로 생성하세요: {context}"
        response = self.llm.invoke(prompt)
        return response.content
    
    