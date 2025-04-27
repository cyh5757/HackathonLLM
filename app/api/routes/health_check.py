from app.api.dto.models import SimpleResponseMessage
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/readiness")
def check_readiness() -> SimpleResponseMessage:
    return SimpleResponseMessage(message="ok")


@router.get("/liveness")
def check_liveness() -> SimpleResponseMessage:
    return SimpleResponseMessage(message="ok")
