from fastapi import APIRouter, HTTPException, Request
from models.analytics import AnalyticsEvent
from services.analytics_service import AnalyticsService
import logging
import json
from pydantic import ValidationError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analytics/track")
async def track_analytics(request: Request):
    try:
        # 요청 데이터 로깅
        body = await request.body()
        body_str = body.decode()
        logger.info(f"Received analytics event: {body_str}")
        
        # JSON 파싱
        try:
            data = json.loads(body_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # 모델 검증
        try:
            event = AnalyticsEvent(**data)
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        
        # 분석 데이터 저장
        result = await AnalyticsService.save_analytics_event(event)
        logger.info(f"Analytics event saved: {event.dict()}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 