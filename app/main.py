from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import dotenv_values

from actions import BaseActions
from schemas import Payload

config = dotenv_values("../.env")
path = config['PATH']
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Payload(BaseModel):
    url: str
    image_id: str


@app.post("/"+path+"/predict")
def predict(payload: Payload):
    return BaseActions.predict(payload)
