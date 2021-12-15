import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
#from decouple import config
from . import actions
from . import schemas


#path = config('path')
# TODO: For local testing, comment out the above lines and uncomment the below line.
# path = 'test'

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/"+path)
# def read_root():
#     return {"Hello": "World"}

class Payload(BaseModel):
    url: str
    image_id: str


@app.post("/test/predict")
def predict(payload: schemas.Payload):

    result = actions.BaseActions.predict(payload)
    return result
