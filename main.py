# uvicorn main:app --reload
from fastapi import FastAPI
from api.recommend_router import router as recommend_router
from api.recommend_detail_router import router as recommend_detail_router
from api.recommend_full_router import router as recommend_full_router


app = FastAPI(title="Stock-Aware Recommendation API")

app.include_router(recommend_router)
app.include_router(recommend_detail_router)
app.include_router(recommend_full_router)
