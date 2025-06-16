from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# MySQL 연결 URL
DATABASE_URL = "mysql+pymysql://moard:moard1234@mysql:3306/moard?charset=utf8mb4"

# Engine, Session, Base 선언
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# FastAPI 의존성으로 사용할 DB 세션 획득 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
