from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings

engine = create_async_engine(str(settings.SQLALCHEMY_DATABASE_URI))
