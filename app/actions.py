from . import schemas
from . import predict
from urllib.request import urlopen

import cv2
import numpy as np

class BaseActions:

    def __init__(self):
        pass

    def preprocessing(img_array: any):
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]
        r_c = clahe.apply(r)
        g_c = clahe.apply(g)
        b_c = clahe.apply(b)
        rgbArray = np.zeros(img_array.shape, 'uint8')
        rgbArray[..., 0] = b_c
        rgbArray[..., 1] = g_c
        rgbArray[..., 2] = r_c

        return rgbArray

    def getImageFromUrl(url: str):
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        ###########
        img_preprocessing = BaseActions.preprocessing(img)
        ##########
        # return img
        return img_preprocessing

    def predict(payload: schemas.Payload):
        frame = BaseActions.getImageFromUrl(payload.url)
        result = predict.run_image(frame, payload.image_id)
        return result
