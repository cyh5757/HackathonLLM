import logging
import sys
from langchain_postgres import PGVector
from app.api.dto.models import Decision_maker, GraphState
from app.repository import pgvector_repository
from app.services.rag_service_test import SNACK_RAG_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from app.core.prompt import rerank_prompt
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import tool
from app.core.config import settings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import TypedDict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_postgres.vectorstores import DistanceStrategy
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

if sys.platform == "win32":
    DB_URI = "postgresql+psycopg://postgres:123123@localhost:5432/test"
else:
    DB_URI = str(settings.SQLALCHEMY_DATABASE_URI)

TABLES = ["snack", "snack_additive"]


def create_agent_executor(
    db_uri: str,
    selected_tables: list[str],
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    verbose: bool = True,
):
    db = SQLDatabase.from_uri(db_uri, include_tables=selected_tables)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=verbose)


agent_executor = create_agent_executor(DB_URI, TABLES)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    embedding_length=1536,
    distance_strategy=DistanceStrategy.COSINE,
    connection=DB_URI,
    logger=logging.getLogger(__name__),
    create_extension=True,
    pre_delete_collection=False,
    use_jsonb=True,
    async_mode=True,
)


async def query_snack_recommendation(
    agent_executor, user_question: str, limit: int = 5
) -> str:
    prompt = f"""
    너는 PostgreSQL 데이터베이스 전문가야.
    사용자 질문: '{user_question}'
    사용자 질문에 따라 적절한 SQL 쿼리를 생성하고 결과를 제공해. limit는 {limit}로 설정해줘.
    단, 'snack' 테이블에서 'barcode', 'snack_type', 'name', 'company', 'total_serving_size', 'allergy_list', 'safe_food_mark_list' 컬럼만 사용해.
    """
    result = await agent_executor.ainvoke({"input": prompt})
    return result["output"]


retriever = vector_store.as_retriever(search_kwargs={"k": 5})


async def vector_search(input: str) -> str:
    # 1. 벡터로 초기 검색
    docs: list[Document] = await retriever.ainvoke(input)
    return "\n---\n".join([doc.page_content for doc in docs])


class Decision_maker(BaseModel):
    reference: str = Field(
        description="Choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material."
    )


class GraphState(TypedDict):
    question: str
    context: list | str
    organize_reference: str


store = {}
decision_maker_output_parser = JsonOutputParser(pydantic_object=Decision_maker)
format_instructions = decision_maker_output_parser.get_format_instructions()

verifier_prompt = PromptTemplate(
    template="""
        You are an expert who verifies the relevance of retrieved data to a given query and organizes the data for LLM to use as final reference material.
        You'll get two types of reference data.
        Choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material.

        <Output format>: MUST answer in string text {format_instructions}
        
        <Question>: {query} 
        <Retrieved data>: {retrieved_data}
        """,
    input_variables=["query", "retrieved_data"],
    partial_variables={"format_instructions": format_instructions},
)


async def decision_maker(state: GraphState) -> GraphState:
    chain = verifier_prompt | llm | decision_maker_output_parser
    verified = await chain.ainvoke(
        {"query": state["question"], "retrieved_data": state["context"]}
    )
    state["organize_reference"] = verified["reference"]
    return state


CONTEXT_WINDOW_LENGTH = 10


def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()

    history = store[session_ids]

    # 최근 N개만 반환하는 슬라이딩 윈도우 로직
    trimmed_history = ChatMessageHistory()
    trimmed_history.messages = history.messages[-CONTEXT_WINDOW_LENGTH:]

    return trimmed_history


async def llm_answer(state: GraphState) -> GraphState:
    prompt = PromptTemplate.from_template(
        """
        You're an helpful assistant who answers questions based on the given context.

        ---Examples---
        #Previous Chat History:
        {chat_history}

        #Question: 
        {question} 

        #Context: 
        {context} 

        #Answer:
        """
    )
    llm_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = prompt | llm_model | StrOutputParser()

    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    input_data = {
        "question": state["question"],
        "chat_history": itemgetter("chat_history"),
        "context": state["context"],
    }

    response = await rag_with_history.ainvoke(
        input_data, config={"configurable": {"session_id": "rag123"}}
    )

    return GraphState(
        answer=response,
        context=state["context"],
        question=state["question"],
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def test_sql_agent():
        print("🧪 [SQL Agent 테스트]")
        question = "알레르기 유발 성분이 없는 감자칩을 알려줘"
        result = await query_snack_recommendation(agent_executor, question)
        print("결과:\n", result)

    async def test_vector_search():
        print("\n🧪 [Vector Search 테스트]")
        query = "아이들이 먹기 안전한 과자 알려줘"
        result = await vector_search(query)
        print("결과:\n", result)

    async def test_vector_search_rerank():
        print("\n🧪 [Vector Search Rerank 테스트]")
        query = "아이들이 먹기 안전한 과자 알려줘"
        result = await vector_search_rerank(query)
        print("결과:\n", result)

    async def test_decision_maker():
        print("\n🧪 [Decision Maker 테스트]")
        test_state = {
            "question": "이 과자가 안전한지 알려줘",
            "context": {
                "sql": "SQL에서 찾은 결과 문서들...",
                "vector": "벡터에서 찾은 결과 문서들...",
            },
            "organize_reference": "",
        }
        updated = await decision_maker(test_state)
        print("선택된 reference:\n", updated["organize_reference"])

    async def test_llm_answer():
        print("\n🧪 [LLM Answer 테스트]")
        state = {
            "question": "이 과자는 알레르기가 없나요?",
            "context": "과자 A는 알레르기 성분이 없습니다. 안전 마크도 있습니다.",
            "organize_reference": "",
        }
        result_state = await llm_answer(state)
        print("답변:\n", result_state["answer"])

    async def test_rerank_comparison():
        print("\n🧪 [ReRank 비교 및 평가 테스트]")
        query = "아이들이 먹기 안전한 과자 알려줘"
        answers = await vector_search_with_comparison(query)
        await evaluate_answers(
            query, answers["original_answer"], answers["reranked_answer"]
        )

    async def main():
        await test_sql_agent()
        await test_vector_search()
        await test_decision_maker()
        await test_llm_answer()

        print("\n🧪 [ReRank 비교 및 평가 테스트]")
        await test_rerank_comparison()

    asyncio.run(main())
