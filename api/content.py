from fastapi import APIRouter, HTTPException
from uuid import uuid4
from models.schema import (
    ContentItem, YouTubeContent, BlogContent, NewsContent
)
from services.supabase_client import supabase

router = APIRouter()

def save_base_content(item: ContentItem) -> str:
    # 1. 중복 확인
    existing = supabase.table("content").select("id").eq("url", item.url).execute()
    if existing.data:
        raise HTTPException(status_code=409, detail="Content already exists.")

    # 2. content 테이블 저장
    content_id = str(uuid4())
    base_data = {
        "id": content_id,
        **item.model_dump(exclude={"id", "created_at"})
    }
    supabase.table("content").insert(base_data).execute()
    return content_id

@router.post("/youtube")
def save_youtube_content(item: ContentItem, details: YouTubeContent):
    content_id = save_base_content(item)
    
    # 유튜브 상세 정보 저장
    youtube_data = details.model_dump()
    youtube_data["content_id"] = content_id
    supabase.table("youtube_content").insert(youtube_data).execute()
    
    return {"status": "ok", "id": content_id}

@router.post("/news")
def save_news_content(item: ContentItem, details: NewsContent):
    content_id = save_base_content(item)
    
    # 뉴스 상세 정보 저장
    news_data = details.model_dump()
    news_data["content_id"] = content_id
    supabase.table("news_content").insert(news_data).execute()
    
    return {"status": "ok", "id": content_id}

@router.post("/blog")
def save_blog_content(item: ContentItem, details: BlogContent):
    content_id = save_base_content(item)
    
    # 블로그 상세 정보 저장
    blog_data = details.model_dump()
    blog_data["content_id"] = content_id
    supabase.table("blog_content").insert(blog_data).execute()
    
    return {"status": "ok", "id": content_id}
