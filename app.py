import uuid, os
from dotenv import load_dotenv
from typing import TypedDict

# from typing import Annotated

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults

## langsmith
from langsmith import Client

# from langchain_teddynote import logging
from langchain_core.tracers.context import collect_runs


from langgraph.graph import END, StateGraph  # Ensure END is imported
from langgraph.checkpoint.memory import MemorySaver  # type: ignore
from langgraph.errors import GraphRecursionError  # type: ignore

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

from langchain.agents import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

## chainlit
import chainlit as cl
from typing import Dict, Optional


# .env 파일 활성화 & API KEY 설정
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
# logging.langsmith("rag_chatbot_test")

llm_4o = ChatOpenAI(model="gpt-4o", temperature=0)


class GraphState(TypedDict):
    question: str  # 질문
    context: list | str  # 문서의 검색 결과
    organize_reference: str  # 문서의 정리된 결과


store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는
def get_session_history(session_ids):
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


############################################
################### tool ###################
#############################################


def create_agent_executor(
    db_uri: str,
    selected_tables: list[str],
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    verbose: bool = True,
):
    """
    SQL Agent Executor를 생성하는 함수.

    Args:
        db_uri (str): PostgreSQL 데이터베이스 URI.
        selected_tables (list[str]): 사용할 테이블 목록.
        model_name (str): 사용할 LLM 모델 이름 (예: gpt-4o, gpt-3.5-turbo 등).
        temperature (float): 모델의 temperature 설정 값.
        verbose (bool): 쿼리 생성 과정의 상세 출력 여부.

    Returns:
        agent_executor: 생성된 SQL Agent Executor 객체.
    """
    # 데이터베이스 연결 설정
    db = SQLDatabase.from_uri(db_uri, include_tables=selected_tables)

    # LLM 모델 설정
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # SQL Agent Executor 생성
    agent_executor = create_sql_agent(
        llm, db=db, agent_type="openai-tools", verbose=verbose
    )

    return agent_executor


def query_snack_recommendation(
    agent_executor, user_question: str, limit: int = 5
) -> str:
    """
    사용자가 입력한 질문을 처리하여 SQL 쿼리를 생성하고 결과를 반환하는 함수.

    Args:
        agent_executor: 생성된 SQL Agent Executor 객체.
        user_question (str): 사용자 질문.
        limit (int): 검색 결과의 최대 개수.

    Returns:
        str: 검색 결과 텍스트.
    """
    prompt = f"""
    너는 PostgreSQL 데이터베이스 전문가야.
    사용자 질문: '{user_question}'
    사용자 질문에 따라 적절한 SQL 쿼리를 생성하고 결과를 제공해. limit는 {limit}로 설정해줘.
    단, 'snack' 테이블에서 'barcode', 'snack_type', 'name', 'company', 'total_serving_size', 'allergy_list', 'safe_food_mark_list' 컬럼만 사용해.
    """

    # 에이전트 실행 및 결과 반환
    result = agent_executor.invoke({"input": prompt})
    return result["output"]


# PostgreSQL 연결 설정

DB_URI = "postgresql+psycopg://postgres:123123@postgres:5432/test"
TABLES = ["snack", "snack_additive"]

# SQL Agent 생성
agent_executor = create_agent_executor(
    db_uri=DB_URI,
    selected_tables=TABLES,
    model_name="gpt-4o",
    temperature=0.0,
    verbose=False,
)


@tool
def sql_search(input):
    """SQL 쿼리 생성기"""
    result_text = query_snack_recommendation(agent_executor, input, limit=5)
    return result_text


@tool
def vector_search(input):
    """벡터 쿼리 생성기"""
    search_tool = TavilySearchResults(max_results=5)
    search_result = search_tool.invoke({"query": input})
    return search_result


# 자료구조 정의 (pydantic)
class Decision_maker(BaseModel):
    reference: str = Field(
        description="Choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material."
    )


# 출력 파서 정의
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


def decision_maker(state: GraphState) -> GraphState:
    chain = verifier_prompt | llm_4o | decision_maker_output_parser
    verified = chain.invoke(
        {"query": state["question"], "retrieved_data": state["context"]}
    )
    state["organize_reference"] = verified["reference"]
    return state


@tool
def decision_maker_A(question, context):
    """Choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material."""
    chain = verifier_prompt | llm_4o | decision_maker_output_parser
    verified = chain.invoke({"query": question, "retrieved_data": context})
    return verified


def llm_answer(state: GraphState) -> GraphState:

    # 프롬프트를 생성합니다.
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

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 프롬프트, 모델, 출력 파서를 체이닝합니다.
    chain = prompt | llm | StrOutputParser()

    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    # 상태에서 질문과 대화 기록을 가져옵니다.
    input_data = {
        "question": state["question"],
        "chat_history": itemgetter("chat_history"),
        "context": state["context"],
    }

    response = rag_with_history.invoke(
        input_data, config={"configurable": {"session_id": "rag123"}}
    )

    return GraphState(
        answer=response,
        context=state["context"],
        question=state["question"],
    )


tools = [sql_search, vector_search]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant, and you use Korean. 
            You have 3 tools to use and `sql_search` and `vector_search` are used to search for relevant documents.
            You can use each of them or both of them.
            After searching, you will get the search results and using `decision_maker` to choose the most relevant reference from multiple sources and organize it for LLM to use as final reference material.
            After that, you will answer the question based on the organized reference.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# agent 함수 수정 - stream 대신 결과를 반환하도록 변경
def agent(input):
    agent = create_tool_calling_agent(llm_4o, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=100,
        max_execution_time=100,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,  # 세션 기록을 가져오는 함수
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    result = agent_with_history.invoke(
        {"input": input}, config={"configurable": {"session_id": "123"}}
    )

    return result


################ chainlit ################


@cl.on_message
async def run_convo(message: cl.Message):
    async with cl.Step(name="langgraph", type="llm") as step:
        step.input = message.content

        user_id = str(uuid.uuid4())

        config = RunnableConfig(
            recursion_limit=20, configurable={"thread_id": user_id, "user_id": user_id}
        )

        inputs = message.content

        try:
            with collect_runs() as cb:
                # agent 함수 호출 및 결과 처리
                result = agent(inputs)

                # 중간 단계 정보 기록
                if "intermediate_steps" in result:
                    for i, step_result in enumerate(result["intermediate_steps"]):
                        step.update(
                            elements=[
                                cl.Text(name=f"Step {i+1}", content=str(step_result))
                            ]
                        )

                answer = result["output"]  # 최종 출력 가져오기

                # 단계별 처리 상태 업데이트
                step.output = answer

        except GraphRecursionError as e:
            print(f"Recursion limit reached: {e}")
            answer = "죄송합니다. 해당 질문에 대해서는 답변할 수 없습니다."
            step.output = answer
        except Exception as e:
            print(f"An error occurred: {e}")
            answer = "죄송합니다. 처리 중 오류가 발생했습니다."
            step.output = answer

    # 최종 응답 전송
    await cl.Message(content=answer).send()
