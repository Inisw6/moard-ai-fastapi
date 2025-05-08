from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.content_router import router as content_router
from api.content import router as content_save_router
from services.content_service_factory import ContentServiceFactory
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(
    title="Content Search API",
    description="YouTube, Naver News, Naver Blog 검색 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
service_factory = ContentServiceFactory()
service_factory.initialize_services(
    youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
    naver_client_id=os.getenv("NAVER_CLIENT_ID"),
    naver_client_secret=os.getenv("NAVER_CLIENT_SECRET")
)

# 라우터 등록
app.include_router(content_router)  # 검색 라우터
app.include_router(content_save_router, prefix="/api")  # 저장 라우터

@app.get("/")
async def root():
    return {"message": "Content Search API is running"}