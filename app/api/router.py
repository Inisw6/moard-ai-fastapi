from fastapi import APIRouter
from app.api.endpoints import predict, embedding, online_learning, model

api_router = APIRouter()

api_router.include_router(
    predict.router, prefix="/predict", tags=["Q-Value Prediction"]
)
api_router.include_router(
    embedding.router, prefix="/embedding", tags=["Content Embedding"]
)
api_router.include_router(
    online_learning.router, prefix="/online-learning", tags=["Online Learning"]
)
api_router.include_router(model.router, prefix="/model", tags=["model"])
