from pydantic import BaseModel
from typing import List, Optional


class NewsRequest(BaseModel):
    topics: List[str]
    source_type: str
