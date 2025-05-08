from fastapi import FastAPI
from api import content

app = FastAPI(
    title="추천 시스템 콘텐츠 API",
    version="1.0.0",
    description="YouTube, 블로그, 뉴스 콘텐츠를 Supabase에 저장하는 API"
)

# 라우터 등록
app.include_router(content.router, prefix="/api", tags=["Content"])