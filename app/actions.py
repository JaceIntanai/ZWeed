from schemas import Payload

class BaseActions():

    def __init__(self):
        pass

    def getImageFromUrl(self, url: str):
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        return img

    def predict(self, payload: Payload):
        # req = urllib.request.urlopen(payload.url)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # img = cv2.imdecode(arr, -1)
        img = self.getImageFromUrl(payload.url)

        cv2.imwrite('dd.png', img)

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
