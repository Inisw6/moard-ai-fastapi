from typing import Dict, Optional
from .youtube_service import YouTubeService
from .naver_news_service import NaverNewsService
from .naver_blog_service import NaverBlogService

class ContentServiceFactory:
    _instance = None
    _services: Dict[str, any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContentServiceFactory, cls).__new__(cls)
        return cls._instance

    def initialize_services(self, 
                          youtube_api_key: str,
                          naver_client_id: str,
                          naver_client_secret: str):
        """
        각 서비스를 초기화합니다.
        """
        self._services['youtube'] = YouTubeService(api_key=youtube_api_key)
        self._services['news'] = NaverNewsService(
            client_id=naver_client_id,
            client_secret=naver_client_secret
        )
        self._services['blog'] = NaverBlogService(
            client_id=naver_client_id,
            client_secret=naver_client_secret
        )

    def get_service(self, service_type: str):
        """
        요청된 타입의 서비스를 반환합니다.
        """
        return self._services.get(service_type)

    @property
    def youtube_service(self) -> Optional[YouTubeService]:
        return self._services.get('youtube')

    @property
    def news_service(self) -> Optional[NaverNewsService]:
        return self._services.get('news')

    @property
    def blog_service(self) -> Optional[NaverBlogService]:
        return self._services.get('blog') 