from typing import List

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
