from datetime import datetime

from sqlmodel import Field, SQLModel


class BaseEntity(SQLModel):
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str
    modified_at: datetime = Field(default_factory=datetime.now)
    modified_by: str
