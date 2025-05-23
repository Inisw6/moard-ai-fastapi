# uvicorn main:app --reload
from fastapi import FastAPI
from api.recommend_router import router as recommend_router

app = FastAPI(title="Stock-Aware Recommendation API")

app.include_router(recommend_router)