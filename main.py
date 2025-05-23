# uvicorn main:app --reload
from fastapi import FastAPI
from app.api.recommend import router as recommend_router

app = FastAPI(title="Stock-Aware Recommendation API")
