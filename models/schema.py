from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Literal
from datetime import datetime
from uuid import UUID

# 1. 콘텐츠 메타 정보 (공통)
class ContentItem(BaseModel):
    id: Optional[UUID] = None
    type: Literal["youtube", "news", "blog"]
    title: str
    url: HttpUrl
    keyword: str
    summary: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    published_at: Optional[datetime] = None
    created_at: Optional[datetime] = None  # Supabase가 자동 채움

# 2. 유튜브 전용 상세 정보
class YouTubeContent(BaseModel):
    content_id: UUID
    channel_title: Optional[str]
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    duration_sec: Optional[int]

# 3. 뉴스 전용 상세 정보
class NewsContent(BaseModel):
    content_id: UUID
    press_name: Optional[str]
    is_opinion: Optional[bool] = False

# 4. 블로그 전용 상세 정보
class BlogContent(BaseModel):
    content_id: UUID
    author: Optional[str]
    sentiment_score: Optional[float]
    word_count: Optional[int]

# 5. 사용자 행동 로그
class LogItem(BaseModel):
    client_id: str
    content_id: UUID
    event_type: Literal["click", "view", "impression"]
    timestamp: Optional[datetime] = None  # 서버에서 자동 처리
    duration_sec: Optional[int] = None
    position: Optional[int] = None

# 6. 추천 결과 기록
class RecommendRecord(BaseModel):
    client_id: str
    content_id: UUID
    algorithm: str
    timestamp: Optional[datetime] = None

# 7. 사용자 프로필 캐시
class UserProfile(BaseModel):
    client_id: str
    last_clicked_ids: Optional[List[UUID]] = None
    preferred_type: Optional[Literal["youtube", "news", "blog"]] = None
    keyword_history: Optional[List[str]] = None
    # embedding은 따로 저장 처리 (binary/vector → Supabase에서는 처리 제외)

# 8. 키워드 태그 연결
class KeywordTag(BaseModel):
    content_id: UUID
    tag: str
