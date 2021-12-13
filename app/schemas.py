from pydantic import BaseModel

# Properties to recieve via API
class Payload(BaseModel):
    url: str
    image_id: str