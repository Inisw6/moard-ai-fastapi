from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models.schema import ContentItem
from services.content_service_factory import ContentServiceFactory

router = APIRouter(prefix="/content", tags=["content"])

@router.get("/search/{keyword}", response_model=List[ContentItem])
async def search_content(
    keyword: str,
    content_type: Optional[str] = None,
    max_results: int = 10
):
    """
    키워드로 콘텐츠를 검색합니다.
    
    - **keyword**: 검색할 키워드
    - **content_type**: 검색할 콘텐츠 타입 (youtube, news, blog, all)
    - **max_results**: 최대 결과 수 (기본값: 10)
    """
    factory = ContentServiceFactory()
    
    if content_type and content_type not in ["youtube", "news", "blog", "all"]:
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    results = []
    
    try:
        if content_type == "youtube" or content_type is None or content_type == "all":
            if youtube_service := factory.youtube_service:
                results.extend(await youtube_service.search(keyword, max_results))
        
        if content_type == "news" or content_type is None or content_type == "all":
            if news_service := factory.news_service:
                results.extend(await news_service.search(keyword, max_results))
        
        if content_type == "blog" or content_type is None or content_type == "all":
            if blog_service := factory.blog_service:
                results.extend(await blog_service.search(keyword, max_results))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 