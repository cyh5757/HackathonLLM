from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SEARCH_SELECT_INSTRUCTIONS = """\
Given:
- A user query.
- A list of search results along with their corresponding indices.
- current date is {datetime}

Task:
1. Evaluate the search results to determine which ones are relevant to the user query.
2. Select and order the indices of the relevant search results in the order of their relevance.
3. If no search result is relevant to the query, return an empty list [].

Your response must be a valid JSON object.
- indices: a list of the selected indices.

Search Result:
{search_results}

User Query:
{query}
"""


class SearchSelectResult(BaseModel):
    indices: List[int] = Field(
        ...,
        description="A list containing the indices of the search results that are relevant to the user query, ordered by relevance. Returns an empty list if no results are relevant."
    )


rerank_prompt = ChatPromptTemplate.from_template(
    """
    사용자 질문: {query}
    문서 내용: {document}

    이 문서가 질문에 답하는 데 얼마나 유용한지를 1~10 점으로 평가하고, 이유도 간단히 설명해줘.
    응답 형식: "점수: <숫자>, 이유: <텍스트>"
    """
)