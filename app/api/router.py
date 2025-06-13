from fastapi import APIRouter
from app.api.endpoints import predict, embedding

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict", tags=["Q-Value Prediction"])
api_router.include_router(embedding.router, prefix="/embedding", tags=["Content Embedding"]) 