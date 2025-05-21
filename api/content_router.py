from fastapi import APIRouter, HTTPException
from typing import List, Optional
from models.content_item import ContentItem
from services.content_service_factory import ContentServiceFactory
from services.content_service import ContentService
from models.content import Content
from datetime import datetime
import uuid
import logging

router = APIRouter(prefix="/content", tags=["content"])
logger = logging.getLogger(__name__)

@router.get("/search/{keyword}", response_model=List[ContentItem])
async def search_content(
    keyword: str,
    content_type: Optional[str] = None,
    max_results: int = 10
):
    """
    키워드로 콘텐츠를 검색하고 결과를 저장합니다.
    
    - **keyword**: 검색할 키워드
    - **content_type**: 검색할 콘텐츠 타입 (youtube, news, blog, all)
    - **max_results**: 최대 결과 수 (기본값: 10)
    """
    factory = ContentServiceFactory()
    
    if content_type and content_type not in ["youtube", "news", "blog", "all"]:
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    results = []
    saved_contents = []
    
    try:
        # 검색 수행
        if content_type == "youtube" or content_type is None or content_type == "all":
            if youtube_service := factory.youtube_service:
                results.extend(await youtube_service.search(keyword, max_results))
        
        if content_type == "news" or content_type is None or content_type == "all":
            if news_service := factory.news_service:
                results.extend(await news_service.search(keyword, max_results))
        
        if content_type == "blog" or content_type is None or content_type == "all":
            if blog_service := factory.blog_service:
                results.extend(await blog_service.search(keyword, max_results))
        
        # 검색 결과를 Content 모델로 변환하여 저장
        for item in results:
            try:
                # URL을 문자열로 변환
                url = str(item.url) if hasattr(item.url, '__str__') else item.url
                
                # thumbnail_url이 없는 경우 None으로 설정
                thumbnail_url = None
                if hasattr(item, 'thumbnail_url'):
                    thumbnail_url = str(item.thumbnail_url) if hasattr(item.thumbnail_url, '__str__') else item.thumbnail_url
                
                content = Content(
                    id=uuid.uuid4(),
                    title=item.title,
                    url=url,
                    type=item.type,
                    stock_keyword=keyword,
                    summary=item.summary,
                    image_url=thumbnail_url,
                    published_at=item.published_at,
                    created_at=datetime.now()
                )
                
                saved_content = await ContentService.save_content(content)
                saved_contents.append(saved_content["data"])
                logger.info(f"Content saved successfully: {content.title}")
            except Exception as e:
                logger.error(f"Error saving content {item.title}: {str(e)}")
                # 저장 실패해도 계속 진행
                continue
        
        if not results:
            logger.warning(f"No results found for keyword: {keyword}")
            return []
        
        # 결과 반환 전에 URL을 문자열로 변환
        return [ContentItem(
            title=item.title,
            url=str(item.url) if hasattr(item.url, '__str__') else item.url,
            type=item.type,
            summary=item.summary,
            thumbnail_url=getattr(item, 'thumbnail_url', None),
            published_at=item.published_at
        ) for item in results]
    
    except Exception as e:
        logger.error(f"Error in search_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 