from fastapi import APIRouter, HTTPException
from models.content import Content
from services.content_service import ContentService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/contents")
async def save_content(content: Content):
    try:
        result = await ContentService.save_content(content)
        logger.info(f"Content saved: {content.dict()}")
        return result
    except Exception as e:
        logger.error(f"Error saving content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/contents/{content_id}")
async def get_content(content_id: str):
    try:
        return await ContentService.get_content(content_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 