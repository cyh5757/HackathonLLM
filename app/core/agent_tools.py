import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from app.core.config import settings
from app.services.rag_agent import (
    query_snack_recommendation,
    agent_executor,
    decision_maker,
    llm_answer,
    vector_search,
    get_session_history,
)

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM 및 벡터 스토어 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    embedding_length=1536,
    distance_strategy=DistanceStrategy.COSINE,
    connection=str(settings.SQLALCHEMY_DATABASE_URI),
    logger=logger,
    create_extension=True,
    pre_delete_collection=False,
    use_jsonb=True,
    async_mode=True,
)
