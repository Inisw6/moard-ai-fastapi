from fastapi import APIRouter, HTTPException
from models.analytics import AnalyticsEvent
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analytics/track")
async def track_analytics(event: AnalyticsEvent):
    try:
        # 여기에 실제 데이터 저장 로직을 구현할 수 있습니다
        # 현재는 로깅만 수행합니다
        logger.info(f"Analytics event received: {event.dict()}")
        
        return {
            "status": "success",
            "message": "Analytics event tracked successfully",
            "data": event.dict()
        }
    except Exception as e:
        logger.error(f"Error tracking analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 