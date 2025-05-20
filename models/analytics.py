from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class AnalyticsEvent(BaseModel):
    client_uid: UUID
    content_id: UUID
    clicked: bool
    dwell_time_seconds: int
    logged_at: datetime 