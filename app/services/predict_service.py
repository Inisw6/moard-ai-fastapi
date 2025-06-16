import json
import logging
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.models.orm import Content, UserLog
from app.services.model_service import ModelService


class PredictService:
    """콘텐츠 Q-value 예측 및 상위 콘텐츠 추천 서비스를 제공하는 클래스.

    Attributes:
        model_service (ModelService): Q-Network 추론용 서비스 인스턴스
    """

    def __init__(self, model_service: ModelService) -> None:
        """PredictService 인스턴스를 초기화합니다.

        Args:
            model_service (ModelService): 모델 추론을 위한 서비스
        """
        self.model_service: ModelService = model_service

    def get_user_embedding(
        self,
        user_id: int,
        db: Session,
        time_decay_factor: float = 0.9,
        max_logs: int = 10,
    ) -> np.ndarray:
        """사용자 상호작용 로그를 기반으로 가중 평균 임베딩을 생성합니다.

        Args:
            user_id (int): 사용자 식별자
            db (Session): 데이터베이스 세션
            time_decay_factor (float): 시간 감쇠 계수
            max_logs (int): 최대로 사용할 로그 수

        Returns:
            np.ndarray: (300,) 크기의 사용자 임베딩 벡터
        """
        logs = (
            db.query(UserLog)
            .filter(UserLog.user_id == user_id)
            .order_by(UserLog.timestamp.desc())
            .limit(max_logs)
            .all()
        )

        if not logs:
            return np.zeros(300, dtype=np.float32)

        now = datetime.now(timezone.utc)
        weighted: List[np.ndarray] = []

        for entry in logs:
            try:
                content = (
                    db.query(Content).filter(Content.id == entry.content_id).first()
                )
                if not content or not content.embedding:
                    continue

                embed_list = json.loads(content.embedding)
                if not isinstance(embed_list, list):
                    continue

                arr = np.array(embed_list, dtype=np.float32)
                ts = entry.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                hours = (now - ts).total_seconds() / 3600
                weight = time_decay_factor**hours

                weighted.append(arr * weight)
            except Exception as err:
                logging.warning(
                    "로그 처리 실패 user_id=%s, content_id=%s: %s",
                    user_id,
                    entry.content_id,
                    err,
                )
                continue

        if not weighted:
            return np.zeros(300, dtype=np.float32)

        combined = np.mean(weighted, axis=0)
        length = combined.shape[0]

        if length < 300:
            combined = np.pad(combined, (0, 300 - length), mode="constant")
        elif length > 300:
            combined = combined[:300]

        return combined.astype(np.float32)

    def get_top_contents(
        self,
        user_id: int,
        content_ids: List[int],
        db: Session,
    ) -> Tuple[List[int], List[float], str]:
        """지정된 콘텐츠 리스트에 대해 Q-value를 예측하고 상위 6개를 반환합니다.

        Args:
            user_id (int): 사용자 식별자
            content_ids (List[int]): 대상 콘텐츠 ID 리스트
            db (Session): 데이터베이스 세션

        Returns:
            Tuple[List[int], List[float], str]:
                상위 6개 콘텐츠 ID 리스트,
                사용자 임베딩 벡터 리스트,
                사용된 모델 파일명

        Raises:
            ValueError: 콘텐츠가 조회되지 않을 경우
        """
        contents = db.query(Content).filter(Content.id.in_(content_ids)).all()
        if not contents:
            raise ValueError("콘텐츠를 찾을 수 없습니다.")

        embeddings: List[List[float]] = []
        for content in contents:
            try:
                emb = json.loads(content.embedding)
                embeddings.append(
                    emb if isinstance(emb, list) else np.random.rand(300).tolist()
                )
            except Exception as err:
                logging.warning("임베딩 파싱 실패 content_id=%s: %s", content.id, err)
                embeddings.append(np.random.rand(300).tolist())

        user_emb = self.get_user_embedding(user_id, db).tolist()
        q_vals = self.model_service.predict(
            user_embedding=user_emb,
            content_embeddings=embeddings,
        )

        pairs = list(zip(contents, q_vals))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top6 = pairs[:6]

        top_ids = [c.id for c, _ in top6]
        model_name = self.model_service.get_model_name()

        return top_ids, user_emb, model_name
