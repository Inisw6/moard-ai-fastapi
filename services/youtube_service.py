from typing import List, Optional
from datetime import datetime
from googleapiclient.discovery import build
from models.schema import ContentItem, YouTubeContent
from .base_service import BaseContentService

class YouTubeService(BaseContentService):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    async def search(self, keyword: str, max_results: int = 10) -> List[ContentItem]:
        """
        YouTube API를 사용하여 비디오를 검색하고 ContentItem 리스트를 반환합니다.
        """
        try:
            # YouTube API 호출
            search_response = self.youtube.search().list(
                q=keyword,
                part='snippet',
                maxResults=max_results,
                type='video'
            ).execute()

            content_items = []
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            # 비디오 상세 정보 가져오기
            if video_ids:
                video_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids)
                ).execute()

                for video in video_response.get('items', []):
                    # 기본 ContentItem 생성
                    content_item = self._create_base_content(
                        title=video['snippet']['title'],
                        url=f"https://www.youtube.com/watch?v={video['id']}",
                        keyword=keyword,
                        content_type="youtube",
                        summary=video['snippet']['description'],
                        image_url=video['snippet']['thumbnails']['high']['url'],
                        published_at=datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    )

                    # YouTube 전용 상세 정보 생성
                    youtube_content = YouTubeContent(
                        content_id=content_item.id,
                        channel_title=video['snippet']['channelTitle'],
                        view_count=int(video['statistics'].get('viewCount', 0)),
                        like_count=int(video['statistics'].get('likeCount', 0)),
                        comment_count=int(video['statistics'].get('commentCount', 0)),
                        duration_sec=self._parse_duration(video['contentDetails']['duration'])
                    )

                    content_items.append(content_item)

            return content_items

        except Exception as e:
            print(f"YouTube API 호출 중 오류 발생: {str(e)}")
            return []

    def _parse_duration(self, duration: str) -> int:
        """
        ISO 8601 형식의 duration을 초 단위로 변환
        """
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds 