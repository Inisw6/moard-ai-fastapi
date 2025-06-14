from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    Text,
    DateTime,
    Float,
    Enum,
    LargeBinary,
    Boolean,
    String,
    ForeignKey,
)
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


# Enum 타입 정의
class EventType(enum.Enum):
    CLICK = "CLICK"
    VIEW = "VIEW"


# Contents 테이블
class Content(Base):
    __tablename__ = "contents"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    description = Column(Text)
    embedding = Column(Text)
    image_url = Column(Text)
    published_at = Column(DateTime)
    query_at = Column(DateTime)
    title = Column(Text)
    type = Column(Text)
    url = Column(Text, unique=True)
    search_query_id = Column(BigInteger, ForeignKey("search_queries.id"))
    search_query = relationship("SearchQuery", back_populates="contents")


# RecommendationContents 조인 테이블
class RecommendationContent(Base):
    __tablename__ = "recommendation_contents"
    content_id = Column(BigInteger, ForeignKey("contents.id"), primary_key=True)
    recommendation_id = Column(
        BigInteger, ForeignKey("recommendations.id"), primary_key=True
    )
    ranks = Column(Integer)


# Recommendations 테이블
class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    flag = Column(Boolean)
    model_version = Column(String(255))
    query = Column(String(255))
    recommended_at = Column(DateTime)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    user = relationship("User", back_populates="recommendations")
    contents = relationship("RecommendationContent", backref="recommendation")


# SearchQueries 테이블
class SearchQuery(Base):
    __tablename__ = "search_queries"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query = Column(String(255))
    searched_at = Column(DateTime)
    stock_info_id = Column(BigInteger, ForeignKey("stock_info.id"))
    stock_info = relationship("StockInfo", back_populates="search_queries")
    contents = relationship("Content", back_populates="search_query")


# StockInfo 테이블
class StockInfo(Base):
    __tablename__ = "stock_info"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    code = Column(String(255))
    industry_detail = Column(String(255))
    industry_type = Column(String(255))
    market_type = Column(String(255))
    name = Column(String(255))
    search_queries = relationship("SearchQuery", back_populates="stock_info")


# StockLogs 테이블
class StockLog(Base):
    __tablename__ = "stock_logs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_name = Column(String(255), nullable=False)
    viewed_at = Column(DateTime, nullable=False)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="stock_logs")


# UserLogs 테이블
class UserLog(Base):
    __tablename__ = "user_logs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(Enum(EventType), nullable=False)
    ratio = Column(Float)
    time = Column(Integer)
    timestamp = Column(DateTime)
    content_id = Column(BigInteger, ForeignKey("contents.id"), nullable=False)
    recommendation_id = Column(BigInteger, ForeignKey("recommendations.id"))
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    content = relationship("Content")
    recommendation = relationship("Recommendation")
    user = relationship("User", back_populates="user_logs")


# Users 테이블
class User(Base):
    __tablename__ = "users"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    uuid = Column(LargeBinary(16))
    recommendations = relationship("Recommendation", back_populates="user")
    stock_logs = relationship("StockLog", back_populates="user")
    user_logs = relationship("UserLog", back_populates="user")
