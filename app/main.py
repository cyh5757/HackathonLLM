from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.api.routes import health_check
from app.core.config import settings


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(health_check.router)

# # CLI 명령어 추가
# if __name__ == "__main__":
#     import asyncio
#     import sys
#     import uvicorn
#     from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
#     from sqlalchemy.orm import sessionmaker
    
#     async def init_db_session():
#         engine = create_async_engine(
#             str(settings.SQLALCHEMY_DATABASE_URI),
#             echo=False,
#             future=True,
#         )
#         async_session = sessionmaker(
#             engine, class_=AsyncSession, expire_on_commit=False
#         )
#         async with async_session() as session:
#             return session
    
#     async def load_data_to_vectordb():
#         from app.repository.snack_vector_loader import load_snack_data_to_vectordb
        
#         print("과자 및 첨가물 데이터를 벡터 DB에 로드하는 중...")
#         session = await init_db_session()
#         try:
#             count = await load_snack_data_to_vectordb(session)
#             print(f"총 {count}개의 문서가 벡터 DB에 저장되었습니다.")
#         finally:
#             await session.close()
    
#     async def load_db_data_to_vectordb():
#         from app.repository.snack_vector_loader import load_db_data_to_vectordb
        
#         print("DB에서 과자 및 첨가물 데이터를 벡터 DB에 로드하는 중...")
#         session = await init_db_session()
#         try:
#             count = await load_db_data_to_vectordb(session)
#             print(f"총 {count}개의 문서가 벡터 DB에 저장되었습니다.")
#         finally:
#             await session.close()
    
#     if len(sys.argv) > 1:
#         command = sys.argv[1]
        
#         if command == "load-snack-data":
#             asyncio.run(load_data_to_vectordb())
#         elif command == "load-db-data":
#             asyncio.run(load_db_data_to_vectordb())
#         else:
#             print(f"알 수 없는 명령어: {command}")
#             print("사용 가능한 명령어: load-snack-data, load-db-data")
#     else:
#         # 일반 서버 실행
#         uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
