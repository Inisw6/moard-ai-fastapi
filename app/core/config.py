import os

from pydantic_settings import BaseSettings


# 네트워크 차원 설정
USER_DIM: int = 300
CONTENT_DIM: int = 300
HIDDEN_DIM: int = 128

# 프로젝트 루트 기준 절대 경로 계산
MODULE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, os.pardir, os.pardir))

# 모델 파일 경로 설정
MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "dqn_model_final.pth")
DOC2VEC_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "doc2vec.model")


class Settings(BaseSettings):
    """
    애플리케이션 환경 설정을 관리합니다.
    """

    PRETRAINED_MODEL_PATH: str = MODEL_PATH
    REDIS_HOST: str = "redis"  # Docker 환경에서 redis 서비스 호스트명
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""


# Pydantic Settings 인스턴스
settings = Settings()
