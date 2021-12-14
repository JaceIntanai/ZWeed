from schemas import Payload
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

    def predict(payload: Payload):
        img = BaseActions.getImageFromUrl(payload.url)
        ##################################
        cv2.imwrite('test.png', img)

        ### example output ###
        return {
            "image_id": payload.image_id,
            "bbox_list": [{
                "category_id": 0,
                "bbox": {
                    "x": 0,
                    "y": 220.66666666666669,
                    "w": 1050.0986882341442,
                    "h": 525.3333333333333
                },
                "score": 0.63508011493555
            }]
        }
