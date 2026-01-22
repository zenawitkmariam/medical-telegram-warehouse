from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TelegramMessage(BaseModel):
    message_id: int
    channel_name: str
    message_date: datetime
    message_text: Optional[str] = None
    has_media: bool = False
    image_path: Optional[str] = None
    views: int = Field(default=0, ge=0)
    forwards: int = Field(default=0, ge=0)

class ChannelData(BaseModel):
    messages: List[TelegramMessage]