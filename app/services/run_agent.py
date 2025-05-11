import asyncio
from app.core.agent_tools import llm
from app.services.rag_agent import (
    query_snack_recommendation,
    agent_executor,
    decision_maker,
    llm_answer,
    vector_search,
    get_session_history,
)
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory


# 🔧 LangChain Tools 정의 (비동기)
@tool
async def sql_search(input: str) -> str:
    """SQL 기반 스낵 추천"""
    return await query_snack_recommendation(agent_executor, input, limit=5)


@tool
async def vector_search_A(input: str) -> str:
    """벡터 기반 검색 (스낵/첨가물 정보)"""
    return await vector_search(input)


@tool
async def decision_maker_A(question: str, context: str) -> str:
    """
    여러 검색 결과에서 핵심 문맥 추출
    """
    state = {
        "question": question,
        "context": context,
        "organize_reference": "",
    }
    result_state = await decision_maker(state)
    return result_state["organize_reference"]


@tool
async def llm_answer_A(question: str, context: str) -> str:
    """
    최종 답변 생성
    """
    state = {
        "question": question,
        "context": context,
        "organize_reference": "",
    }
    result_state = await llm_answer(state)
    return result_state["answer"]


# 🧠 Agent Prompt 설정
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant, and you use Korean.
        You have tools: sql_search, vector_search_A, decision_maker_A, llm_answer_A.
        Use search tools first, summarize with decision_maker_A, and finish with llm_answer_A.""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# 🧠 Agent 생성 함수 (비동기)
# 전체 에이전트 실행을 담당하는 비동기 함수
async def run_agent_full(input_text: str):
    # LangChain의 create_tool_calling_agent를 사용하여 Tool 기반 LLM Agent 생성
    # 이 Agent는 내부적으로 사용자가 정의한 Tool 들을 호출할 수 있음
    agent = create_tool_calling_agent(
        llm,
        tools=[
            sql_search,
            vector_search_A,
            decision_maker_A,
            llm_answer_A,
        ],  # 정의한 Tool 목록
        prompt=agent_prompt,  # 에이전트가 따를 프롬프트 템플릿
    )

    # AgentExecutor는 위에서 만든 agent를 실행할 수 있는 실행기
    executor = AgentExecutor(
        agent=agent,
        tools=[
            sql_search,
            vector_search_A,
            decision_maker_A,
            llm_answer_A,
        ],  # tool들을 agent에게 다시 연결
        verbose=True,  # 실행 과정 로그 출력
        max_iterations=10,  # 최대 반복 횟수 (tool 사용 반복)
        handle_parsing_errors=True,  # LLM 출력이 파싱 안 되면 자동 복구 시도
        return_intermediate_steps=False,  # 중간 tool 결과를 반환하지 않음
    )

    # 대화 기록을 유지하기 위한 메시지 히스토리 래퍼
    agent_with_history = RunnableWithMessageHistory(
        executor,  # 위에서 정의한 AgentExecutor
        get_session_history,  # 세션 ID에 따라 대화 이력을 저장하는 함수
        input_messages_key="input",  # 사용자 입력 키
        history_messages_key="chat_history",  # 기록될 히스토리 키
    )

    # 에이전트 실행 (입력값과 세션ID 포함)
    result = await agent_with_history.ainvoke(
        {"input": input_text},
        config={"configurable": {"session_id": "rag123"}},  # 세션 ID를 넘김
    )

    # 최종 응답 반환
    return result


# 🚀 실행
if __name__ == "__main__":

    async def main():
        user_input = "어린이에게 좋은 과자를 추천해줘"
        result = await run_agent_full(user_input)
        print("최종 결과:\n", result["output"])

    # agent tools
    # prompt chain : prompt split
    # routings : agentic loop
    # parelllization : master + worker
    # context window lenght : 5
    # agentic loop :
    asyncio.run(main())
