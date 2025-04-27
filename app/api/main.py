from fastapi import APIRouter

from app.api.routes import snacks

api_router = APIRouter()

api_router.include_router(snacks.router)

# if settings.ENVIRONMENT == "local":
#     api_router.include_router(private.router)
