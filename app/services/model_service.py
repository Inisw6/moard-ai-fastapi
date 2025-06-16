import os
from typing import List, Optional
import logging

import torch
from torch import Tensor

from app.ml.q_network import DuelingQNetwork
from app.core.config import USER_DIM, CONTENT_DIM, HIDDEN_DIM, MODEL_PATH


class ModelService:
    """
    싱글턴 패턴으로 모델 로딩 및 추론 기능을 관리하는 서비스 클래스.

    Attributes:
        model (Optional[torch.nn.Module]): 로드된 Q-Network 모델 인스턴스
    """

    _instance: "ModelService" = None  # type: ignore[name-defined]
    model: Optional[torch.nn.Module] = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        """설정된 경로에서 사전 학습된 모델을 로드합니다."""
        network = DuelingQNetwork(
            user_dim=USER_DIM,
            content_dim=CONTENT_DIM,
            hidden_dim=HIDDEN_DIM,
        )

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            network.load_state_dict(checkpoint["q_net_state"])
            network.eval()
            self.model = network
            logging.info("모델 로딩 완료: %s", MODEL_PATH)
        except FileNotFoundError:
            logging.error("모델 파일을 찾을 수 없습니다: %s", MODEL_PATH)
            self.model = None
        except Exception as err:
            logging.error("모델 로딩 중 오류 발생: %s", err)
            self.model = None

    def get_model_name(self) -> str:
        """현재 로드된 모델의 파일명을 반환합니다.

        Returns:
            str: 모델 파일명 또는 로드 실패 메시지
        """
        if self.model is None:
            return "No model loaded"
        return os.path.basename(MODEL_PATH)

    def predict(
        self,
        user_embedding: List[float],
        content_embeddings: List[List[float]],
    ) -> List[float]:
        """사용자 및 콘텐츠 임베딩으로부터 Q-value 리스트를 예측합니다.

        Args:
            user_embedding (List[float]): 사용자 임베딩 벡터
            content_embeddings (List[List[float]]): 콘텐츠 임베딩 벡터 리스트

        Returns:
            List[float]: 예측된 Q-value 리스트

        Raises:
            RuntimeError: 모델이 로드되지 않았거나 추론 오류 발생 시
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        try:
            user_tensor: Tensor = torch.tensor([user_embedding], dtype=torch.float32)
            content_tensor: Tensor = torch.tensor(
                content_embeddings, dtype=torch.float32
            )

            num_items = content_tensor.size(0)
            user_expanded: Tensor = user_tensor.expand(num_items, -1)

            with torch.no_grad():
                q_values: Tensor = self.model(user_expanded, content_tensor)

            return q_values.flatten().tolist()
        except Exception as err:
            logging.error("추론 중 오류 발생: %s", err)
            raise RuntimeError("추론 과정에서 오류가 발생했습니다.") from err


def get_model_service() -> ModelService:
    """FastAPI 의존성 주입용 함수로 ModelService 싱글턴을 반환합니다.

    Returns:
        ModelService: 모델 서비스 인스턴스
    """
    return ModelService()
