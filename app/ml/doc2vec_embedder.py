import logging
import re
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from joblib import Parallel, delayed

from app.core.config import DOC2VEC_MODEL_PATH
from app.services.content_service import get_contents


class Doc2VecContentEmbedder:
    """Doc2Vec 모델을 사용하여 텍스트 기반 콘텐츠를 임베딩합니다.

    이 클래스는 사전 학습된 Doc2Vec 모델을 로드하여 콘텐츠의 제목과 설명을
    바탕으로 고정된 차원의 벡터를 생성합니다.

    Attributes:
        doc2vec_model (Doc2Vec): 로드된 Gensim Doc2Vec 모델.
        pretrained_dim (int): Doc2Vec 모델의 벡터 크기.
        content_dim (int): 임베더의 최종 출력 차원.
        all_contents_df (pd.DataFrame): 모든 콘텐츠 정보가 담긴 DataFrame.
        content_types (list[str]): 데이터에 존재하는 모든 콘텐츠 타입 리스트.
    """

    def __init__(
        self,
        model_path: str = DOC2VEC_MODEL_PATH,
        content_dim: int = 300,
        all_contents_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Doc2VecContentEmbedder 인스턴스를 초기화합니다.

        Args:
            model_path (str, optional): Doc2Vec 모델 파일의 경로.
                기본값은 config 파일에 정의된 경로입니다.
            content_dim (int, optional): 원하는 출력 임베딩 차원. Doc2Vec 모델의
                벡터 크기와 다를 경우, 모델의 크기를 따릅니다. 기본값은 300.
            all_contents_df (Optional[pd.DataFrame], optional): 외부에서 주입할
                콘텐츠 DataFrame. 주입되지 않으면 `get_contents()`를 호출하여
                DB에서 직접 가져옵니다. 기본값은 None.
        """
        try:
            logging.info("Doc2Vec 모델 '%s' 로딩 중...", model_path)
            self.doc2vec_model = Doc2Vec.load(model_path)
        except FileNotFoundError:
            logging.error("Doc2Vec 모델을 찾을 수 없습니다: %s", model_path)
            raise

        self.pretrained_dim = self.doc2vec_model.vector_size
        self.content_dim = self.pretrained_dim

        if content_dim != self.pretrained_dim:
            logging.warning(
                "설정된 content_dim(%d)이 Doc2Vec vector_size(%d)와 다릅니다. "
                "Doc2Vec의 벡터 크기를 따릅니다.",
                content_dim,
                self.pretrained_dim,
            )

        self.all_contents_df = (
            all_contents_df if all_contents_df is not None else get_contents()
        )
        if not self.all_contents_df.empty:
            self.content_types = self.all_contents_df["type"].unique().tolist()
        else:
            self.content_types = ["youtube", "blog", "news"]

    def output_dim(self) -> int:
        """콘텐츠 임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터의 차원.
        """
        return self.content_dim

    def embed_content(self, content: dict) -> np.ndarray:
        """주어진 콘텐츠를 Doc2Vec을 사용하여 임베딩합니다.

        Args:
            content (dict): 'title'과 'description' 키를 포함하는 콘텐츠 딕셔너리.

        Returns:
            np.ndarray: `(content_dim,)` 크기의 임베딩 벡터. 텍스트가 비어있거나
                추론에 실패하면 제로 벡터를 반환합니다.
        """
        raw_text = content.get("title", "") + " " + content.get("description", "")
        clean_text = re.sub(r"<.*?>", "", raw_text).strip()

        if not clean_text:
            tokens = []
        else:
            tokens = clean_text.split()

        try:
            inferred_vec = self.doc2vec_model.infer_vector(tokens)
            return np.array(inferred_vec, dtype=np.float32)
        except Exception as e:
            logging.warning("Doc2Vec 추론 실패: %s. 제로 벡터를 반환합니다.", e)
            return np.zeros(self.pretrained_dim, dtype=np.float32)

    def embed_contents(self, contents: List[Dict]) -> np.ndarray:
        """주어진 여러 콘텐츠를 병렬로 임베딩합니다.

        Args:
            contents (List[Dict]): 각각 'title'과 'description' 키를 포함하는
                콘텐츠 딕셔너리의 리스트.

        Returns:
            np.ndarray: `(num_contents, content_dim)` 크기의 임베딩 벡터 배열.
        """
        if not contents:
            return np.array([], dtype=np.float32).reshape(0, self.content_dim)

        # joblib을 사용하여 embed_content 함수를 병렬로 실행합니다.
        # n_jobs=-1은 사용 가능한 모든 CPU 코어를 사용하라는 의미입니다.
        embeddings = Parallel(n_jobs=-1)(
            delayed(self.embed_content)(content) for content in contents
        )

        return np.array(embeddings, dtype=np.float32)
