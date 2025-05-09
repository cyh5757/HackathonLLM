import logging

from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    embedding_length=1536,
    distance_strategy=DistanceStrategy.COSINE,
    connection=str(settings.SQLALCHEMY_DATABASE_URI),
    logger=logging.getLogger(__name__),
    create_extension=True,  # "CREATE EXTENSION IF NOT EXISTS vector;" 실행
    pre_delete_collection=False,  # 기존 테이블 데이터(레코드) 삭제
    use_jsonb=True,
    async_mode=True,
)
