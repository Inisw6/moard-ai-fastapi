from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# MySQL 연결 URL
DATABASE_URL: str = "mysql+pymysql://moard:moard1234@mysql:3306/moard?charset=utf8mb4"

# SQLAlchemy 엔진 및 세션 구성
engine: Engine = create_engine(DATABASE_URL, echo=True)
SessionLocal: sessionmaker = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI 의존성으로 사용할 DB 세션을 생성하고, 요청 종료 시 세션을 종료합니다.

    Yields:
        Session: 데이터베이스 세션 인스턴스
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
