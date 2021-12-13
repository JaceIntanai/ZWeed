
import numpy as np

from actions import BaseActions
from schemas import Payload
from urllib.parse import urlparse
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import dotenv_values



config = dotenv_values(".env")
app = FastAPI()


@app.get("/")
def index():
    return {"Hello": "World!!"}

@app.post("/"+path+"/predict")
def predict(payload: Payload):
    return BaseActions.predict(payload)


# class Payload(BaseModel):
#     url: str
#     image_id: str


# @app.post("/"+path+"/predict")
# def predict(payload: Payload):
#     req = urllib.request.urlopen(payload.url)
#     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     img = cv2.imdecode(arr, -1)

#     cv2.imwrite('dd.png', img)

#     return {
#         "image_id": payload.image_id,
#         "bbox_list": [{
#             "category_id": 0,
#             "bbox": {
#                 "x": 0,
#                 "y": 220.66666666666669,
#                 "w": 1050.0986882341442,
#                 "h": 525.3333333333333
#             },
#             "score": 0.63508011493555
#         }]
#     }
