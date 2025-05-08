from abc import ABC, abstractmethod
from typing import List, Optional
from models.schema import ContentItem
from datetime import datetime
from uuid import uuid4

class BaseContentService(ABC):
    def __init__(self):
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None

    @abstractmethod
    async def search(self, keyword: str, max_results: int = 10) -> List[ContentItem]:
        """
        키워드로 콘텐츠를 검색하는 기본 메서드
        """
        pass

    def _create_base_content(self, 
                           title: str, 
                           url: str, 
                           keyword: str,
                           content_type: str,
                           summary: Optional[str] = None,
                           image_url: Optional[str] = None,
                           published_at: Optional[datetime] = None) -> ContentItem:
        """
        기본 ContentItem 객체를 생성하는 헬퍼 메서드
        """
        return ContentItem(
            id=uuid4(),
            type=content_type,
            title=title,
            url=url,
            keyword=keyword,
            summary=summary,
            image_url=image_url,
            published_at=published_at,
            created_at=datetime.utcnow()
        ) 