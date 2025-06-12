def load_all_ml_models():
    """애플리케이션에서 사용하는 모든 ML 모델/임베더 모듈을 임포트합니다.

    이 함수는 각 모듈에 정의된 `@register` 데코레이터가 실행되도록 하여,
    해당 모델/임베더가 레지스트리에 등록되게 합니다.
    """
    from . import doc2vec_embedder

    print("ML 모델 및 임베더가 성공적으로 로드 및 등록되었습니다.") 