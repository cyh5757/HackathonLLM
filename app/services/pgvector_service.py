import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from app.api.deps import SessionDep
from app.api.dto.models import DocumentRes
from app.core import agent_tools
from app.core.prompt import SearchSelectResult, SEARCH_SELECT_INSTRUCTIONS
from app.repository import pgvector_repository
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def search_pgvector(
    session: SessionDep,  # DB session
    query: str,
) -> list[DocumentRes] | None:
    similar_memory_list: list[tuple[Document, float]] = await pgvector_repository.search_similar_memories(
        query=query,
        key_prefix="sample_key",
        field_name="my_data",
        size=10
    )

    result: list[tuple[Document, float]] = await _select_relevant_search_results_by_llm(
        search_results=similar_memory_list,
        query=query,
    )

    res: list[DocumentRes] = [DocumentRes.from_document(x[0]) for x in result]

    return res


async def _select_relevant_search_results_by_llm(
    search_results: list[tuple[Document, float]],
    query: str,
) -> list[tuple[Document, float]]:
    formatted_results = "\n".join(
        [
            f"Index {i}:\n{doc.page_content}{doc.metadata}\n"
            for i, (doc, _) in enumerate(search_results)
        ]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant who selects relevant search results based on the user query."),
        ("human", SEARCH_SELECT_INSTRUCTIONS.format(
            datetime=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%A, %B %d, %Y, %I:%M %p"),
            search_results=formatted_results,
            query=query
        ))
    ])

    structured_llm = prompt | agent_tools.llm.with_structured_output(SearchSelectResult)
    result: SearchSelectResult = await structured_llm.ainvoke({})

    return [search_results[i] for i in result.indices if i < len(search_results)]
