import os

# 모델 관련 설정을 정의합니다.
# 하이퍼파라미터, 모델 파일 경로 등을 포함합니다.
USER_DIM = 30
CONTENT_DIM = 5
HIDDEN_DIM = 128

# 프로젝트 루트 디렉터리 기준의 절대 경로를 생성하여
# 어느 위치에서 실행해도 일관된 경로를 유지합니다.
MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "dqn_model_seed0_final.pth"
    )
)
DOC2VEC_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "doc2vec.model")
)

"""애플리케이션 설정을 관리하는 모듈."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정.

    Attributes:
        PRETRAINED_MODEL_PATH: 사전 학습된 모델 파일 경로
        REDIS_HOST: Redis 서버 호스트
        REDIS_PORT: Redis 서버 포트
        REDIS_DB: Redis 데이터베이스 번호
        REDIS_PASSWORD: Redis 비밀번호
    """

    PRETRAINED_MODEL_PATH: str = "models/dqn_model_seed0_final.pth"

    # Redis 설정
    REDIS_HOST: str = "localhost"  # Docker 환경에서는 "redis"로 변경
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""


settings = Settings()
