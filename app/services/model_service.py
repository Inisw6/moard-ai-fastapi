import torch
from app.ml.q_network import QNetwork
from app.core import config

class ModelService:
    """모델 로딩 및 추론을 관리하는 싱글턴 서비스 클래스."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """설정 파일에 지정된 경로에서 모델을 로드하여 인스턴스 변수에 저장합니다."""
        model = QNetwork(
            user_dim=config.USER_DIM,
            content_dim=config.CONTENT_DIM,
            hidden_dim=config.HIDDEN_DIM,
        )
        try:
            # CPU 환경에서 모델을 로드합니다.
            checkpoint = torch.load(config.MODEL_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["q_net_state"])
            model.eval()
            self.model = model
            print("모델 로딩 완료.")
        except FileNotFoundError:
            print(f"Error: 모델 파일을 찾을 수 없습니다. 경로: {config.MODEL_PATH}")
            self.model = None
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            self.model = None

    def predict(self, user_embedding: list[float], content_embeddings: list[list[float]]) -> list[float]:
        """주어진 임베딩으로 Q-value를 예측합니다.

        Args:
            user_embedding (list[float]): 단일 사용자의 임베딩 벡터.
            content_embeddings (list[list[float]]): 여러 콘텐츠의 임베딩 벡터 리스트.

        Returns:
            list[float]: 각 콘텐츠에 대한 예측된 Q-value 리스트.

        Raises:
            RuntimeError: 모델이 성공적으로 로드되지 않았거나 추론 중 오류가 발생한 경우.
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 서버 로그를 확인하세요.")

        try:
            # 입력 데이터를 Tensor로 변환합니다.
            user_tensor = torch.tensor([user_embedding], dtype=torch.float32)
            content_tensor = torch.tensor(content_embeddings, dtype=torch.float32)

            # 모든 콘텐츠 임베딩에 대해 동일한 사용자 임베딩을 사용하기 위해 확장합니다.
            num_contents = content_tensor.shape[0]
            user_tensor_expanded = user_tensor.expand(num_contents, -1)

            # 그래디언트 계산을 중단하고 추론을 실행합니다.
            with torch.no_grad():
                q_values = self.model(user_tensor_expanded, content_tensor)

            return q_values.flatten().tolist()
        except Exception as e:
            print(f"추론 중 오류 발생: {e}")
            raise RuntimeError("추론 과정에서 오류가 발생했습니다.")


def get_model_service() -> ModelService:
    """ModelService의 싱글턴 인스턴스를 반환합니다.

    FastAPI의 의존성 주입 시스템에서 사용됩니다.

    Returns:
        ModelService: 모델 서비스의 싱글턴 인스턴스.
    """
    return ModelService() 