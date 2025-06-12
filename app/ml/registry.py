from typing import Callable, Dict, Type, Any
from app.ml.base_embedder import BaseContentEmbedder

# 임베더 클래스를 저장하기 위한 글로벌 레지스트리
EMBEDDER_REGISTRY: Dict[str, Type[BaseContentEmbedder]] = {}


def register(name: str) -> Callable[[Type[BaseContentEmbedder]], Type[BaseContentEmbedder]]:
    """임베더 클래스를 레지스트리에 등록하는 데코레이터.

    Example:
        @register("my_embedder")
        class MyEmbedder(BaseContentEmbedder):
            ...

    Args:
        name (str): 레지스트리에 등록할 임베더의 이름.

    Returns:
        Callable: 클래스를 받아 등록을 수행하고 그대로 반환하는 데코레이터.
    """

    def decorator(cls: Type[BaseContentEmbedder]) -> Type[BaseContentEmbedder]:
        """주어진 클래스를 레지스트리에 등록합니다."""
        if name in EMBEDDER_REGISTRY:
            raise ValueError(f"오류: '{name}' 이름으로 등록된 임베더가 이미 존재합니다.")
        EMBEDDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_embedder(name: str, **kwargs: Any) -> BaseContentEmbedder:
    """레지스트리에서 이름으로 임베더 인스턴스를 생성하고 반환합니다.

    Args:
        name (str): 가져올 임베더의 이름.
        **kwargs: 임베더의 `__init__` 메서드에 전달될 키워드 인수.

    Returns:
        BaseContentEmbedder: 지정된 이름의 임베더 인스턴스.

    Raises:
        ValueError: 해당 이름의 임베더가 레지스트리에 없는 경우.
    """
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(f"오류: '{name}' 이름의 임베더를 찾을 수 없습니다. "
                         f"사용 가능한 임베더: {list(EMBEDDER_REGISTRY.keys())}")
    embedder_class = EMBEDDER_REGISTRY[name]
    return embedder_class(**kwargs) 