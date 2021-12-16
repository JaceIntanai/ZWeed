from . import schemas
from . import predict
from urllib.request import urlopen

import cv2
import numpy as np

class BaseActions:

    def __init__(self):
        pass

    def getImageFromUrl(url: str):
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img

    def predict(payload: schemas.Payload):
        frame = BaseActions.getImageFromUrl(payload.url)
        result = predict.run_image(frame, payload.image_id)
        return result
