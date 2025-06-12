import os

# 모델 관련 설정을 정의합니다.
# 하이퍼파라미터, 모델 파일 경로 등을 포함합니다.
USER_DIM = 30
CONTENT_DIM = 5
HIDDEN_DIM = 128

# 프로젝트 루트 디렉터리 기준의 절대 경로를 생성하여
# 어느 위치에서 실행해도 일관된 경로를 유지합니다.
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "dqn_model_seed0_final.pth")
)
DOC2VEC_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "doc2vec.model")
) 