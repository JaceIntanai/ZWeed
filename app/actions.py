from . import schemas
from . import predict
from urllib.request import urlopen

import cv2
import numpy as np

class BaseActions:

    def __init__(self):
        pass

    def preprocessing(img_array: any):
        print(img_array)
        # image = cv2.imread('all_datasets/20201110122206408.png')
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]
        r_c = clahe.apply(r)
        g_c = clahe.apply(g)
        b_c = clahe.apply(b)
        rgbArray = np.zeros(img_array.shape, 'uint8')
        rgbArray[..., 0] = r_c
        rgbArray[..., 1] = g_c
        rgbArray[..., 2] = b_c
        print('----start----')
        print('1111 rgbArray ==> ', rgbArray)
        print('----')
        # cv2.imwrite('rgb.png', rgbArray)
        cv2.imwrite('/Users/mai/arv-hackathon/code/ZWeed/sample_out_1.png', rgbArray)

        return rgbArray

    def getImageFromUrl(url: str):
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        ###########
        img_preprocessing = BaseActions.preprocessing(img)
        cv2.imwrite('/Users/mai/arv-hackathon/code/ZWeed/sample_out_preprocessing.png', img_preprocessing)
        ##########
        # return img
        return img_preprocessing

    def predict(payload: schemas.Payload):
        frame = BaseActions.getImageFromUrl(payload.url)
        result = predict.run_image(frame, payload.image_id)
        return result
