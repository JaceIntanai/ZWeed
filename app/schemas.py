from pydantic import BaseModel

class Payload(BaseModel):
    url: str
    image_id: str