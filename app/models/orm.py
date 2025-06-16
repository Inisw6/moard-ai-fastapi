import enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class EventType(enum.Enum):
    """
    사용자 상호작용 이벤트 유형을 나타내는 열거형입니다.

    Attributes:
        CLICK: 콘텐츠 클릭 이벤트
        VIEW: 콘텐츠 조회 이벤트
    """

    CLICK = "CLICK"
    VIEW = "VIEW"


class Content(Base):
    """
    콘텐츠 정보를 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 콘텐츠 고유 식별자
        title (Text): 콘텐츠 제목
        description (Text): 콘텐츠 설명
        embedding (Text): 임베딩 벡터 (직렬화)
        image_url (Text): 콘텐츠 이미지 URL
        url (Text): 콘텐츠 원본 URL (고유)
        type (Text): 콘텐츠 타입
        published_at (DateTime): 게시 일시
        query_at (DateTime): 검색 일시
        search_query_id (BigInteger): 연관 검색 쿼리 ID
    Relationships:
        search_query: 검색 쿼리와의 외래키 관계
    """

    __tablename__ = "contents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(Text)
    description = Column(Text)
    embedding = Column(Text)
    image_url = Column(Text)
    url = Column(Text, unique=True)
    type = Column(Text)
    published_at = Column(DateTime)
    query_at = Column(DateTime)
    search_query_id = Column(BigInteger, ForeignKey("search_queries.id"))
    search_query = relationship("SearchQuery", back_populates="contents")


class RecommendationContent(Base):
    """
    추천 결과와 콘텐츠 간의 다대다 관계를 나타내는 조인 테이블입니다.

    Columns:
        content_id (BigInteger): 콘텐츠 ID (외래키)
        recommendation_id (BigInteger): 추천 ID (외래키)
        ranks (Integer): 추천 순위
    """

    __tablename__ = "recommendation_contents"

    content_id = Column(BigInteger, ForeignKey("contents.id"), primary_key=True)
    recommendation_id = Column(
        BigInteger, ForeignKey("recommendations.id"), primary_key=True
    )
    ranks = Column(Integer)


class Recommendation(Base):
    """
    사용자별 추천 기록을 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 추천 고유 식별자
        user_id (BigInteger): 사용자 ID (외래키)
        query (String): 추천 시 사용된 질의
        embedding (Text): 추천 임베딩 벡터 (직렬화)
        model_version (String): 사용된 모델 버전
        recommended_at (DateTime): 추천 생성 시각
        flag (Boolean): 추천 활성화 여부
    Relationships:
        user: User와의 외래키 관계
        contents: RecommendationContent 관계 목록
    """

    __tablename__ = "recommendations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    query = Column(String(255))
    embedding = Column(Text)
    model_version = Column(String(255))
    recommended_at = Column(DateTime)
    flag = Column(Boolean)

    user = relationship("User", back_populates="recommendations")
    contents = relationship("RecommendationContent", backref="recommendation")


class SearchQuery(Base):
    """
    사용자 검색 쿼리 이력을 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 검색 고유 식별자
        query (String): 검색어
        searched_at (DateTime): 검색 시각
        stock_info_id (BigInteger): 관련 주식 정보 ID
    Relationships:
        stock_info: StockInfo와의 외래키 관계
        contents: Content 목록
    """

    __tablename__ = "search_queries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query = Column(String(255))
    searched_at = Column(DateTime)
    stock_info_id = Column(BigInteger, ForeignKey("stock_info.id"))

    stock_info = relationship("StockInfo", back_populates="search_queries")
    contents = relationship("Content", back_populates="search_query")


class StockInfo(Base):
    """
    주식 정보를 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 주식 고유 식별자
        code (String): 주식 코드
        name (String): 주식 이름
        market_type (String): 시장 유형
        industry_type (String): 업종 유형
        industry_detail (String): 업종 상세
    Relationships:
        search_queries: SearchQuery 목록
    """

    __tablename__ = "stock_info"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    code = Column(String(255))
    name = Column(String(255))
    market_type = Column(String(255))
    industry_type = Column(String(255))
    industry_detail = Column(String(255))

    search_queries = relationship("SearchQuery", back_populates="stock_info")


class StockLog(Base):
    """
    사용자의 주식 조회 이력을 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 로그 고유 식별자
        user_id (BigInteger): 사용자 ID
        stock_name (String): 조회한 주식 이름
        viewed_at (DateTime): 조회 시각
    Relationships:
        user: User와의 외래키 관계
    """

    __tablename__ = "stock_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    stock_name = Column(String(255), nullable=False)
    viewed_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="stock_logs")


class UserLog(Base):
    """
    사용자 상호작용 로그를 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 로그 고유 식별자
        user_id (BigInteger): 사용자 ID
        event_type (Enum): 상호작용 유형
        content_id (BigInteger): 관련 콘텐츠 ID
        recommendation_id (BigInteger): 관련 추천 ID
        ratio (Float): 사용 비율
        time (Integer): 상호작용 시간
        timestamp (DateTime): 로그 생성 시각
    Relationships:
        user: User와의 외래키 관계
        content: Content 참조
        recommendation: Recommendation 참조
    """

    __tablename__ = "user_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    event_type = Column(Enum(EventType), nullable=False)
    content_id = Column(BigInteger, ForeignKey("contents.id"), nullable=False)
    recommendation_id = Column(BigInteger, ForeignKey("recommendations.id"))
    ratio = Column(Float)
    time = Column(Integer)
    timestamp = Column(DateTime)

    user = relationship("User", back_populates="user_logs")
    content = relationship("Content")
    recommendation = relationship("Recommendation")


class User(Base):
    """
    사용자 정보를 저장하는 테이블입니다.

    Columns:
        id (BigInteger): 사용자 고유 식별자
        uuid (LargeBinary): 사용자 UUID
    Relationships:
        recommendations: Recommendation 목록
        stock_logs: StockLog 목록
        user_logs: UserLog 목록
    """

    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    uuid = Column(LargeBinary(16))

    recommendations = relationship("Recommendation", back_populates="user")
    stock_logs = relationship("StockLog", back_populates="user")
    user_logs = relationship("UserLog", back_populates="user")
