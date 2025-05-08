from typing import List, Optional
from datetime import datetime
import aiohttp
from urllib.parse import urlencode
from models.schema import ContentItem, NewsContent
from .base_service import BaseContentService

class NaverNewsService(BaseContentService):
    def __init__(self, client_id: str, client_secret: str):
        super().__init__()
        self.api_key = client_id
        self.api_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/news.json"

    async def search(self, keyword: str, max_results: int = 10) -> List[ContentItem]:
        """
        네이버 뉴스 API를 사용하여 뉴스를 검색하고 ContentItem 리스트를 반환합니다.
        """
        try:
            headers = {
                "X-Naver-Client-Id": self.api_key,
                "X-Naver-Client-Secret": self.api_secret
            }
            
            # URL 파라미터 직접 구성
            query_params = {
                "query": keyword,
                "display": str(max_results),
                "sort": "sim"
            }
            url = f"{self.base_url}?{urlencode(query_params)}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        print(f"네이버 뉴스 API 호출 실패: {response.status}")
                        return []

                    data = await response.json()
                    content_items = []

                    for item in data.get('items', []):
                        # HTML 태그 제거
                        title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                        description = item.get('description', '').replace('<b>', '').replace('</b>', '')
                        link = item.get('link', '')
                        pub_date = item.get('pubDate', '')
                        publisher = item.get('publisher', '')

                        if not all([title, link, pub_date]):
                            continue

                        try:
                            # 기본 ContentItem 생성
                            content_item = self._create_base_content(
                                title=title,
                                url=link,
                                keyword=keyword,
                                content_type="news",
                                summary=description,
                                published_at=datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                            )

                            # 뉴스 전용 상세 정보 생성
                            news_content = NewsContent(
                                content_id=content_item.id,
                                press_name=publisher if publisher else None,
                                is_opinion=False  # 기본값
                            )

                            content_items.append(content_item)
                        except (ValueError, KeyError) as e:
                            print(f"뉴스 아이템 처리 중 오류 발생: {str(e)}")
                            continue

                    return content_items

        except Exception as e:
            print(f"네이버 뉴스 API 호출 중 오류 발생: {str(e)}")
            return [] 