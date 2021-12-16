import cv2, numpy as np, os

current_dir = os.getcwd()
cfg = os.path.join("yolov4-tiny_training.cfg")
weights = os.path.join("yolov4-tiny_training_last.weights")

yolo = cv2.dnn.readNetFromDarknet(cfg, weights)

classes = ["pipe", "corner", "flange", "anode"]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# fps = None

def run_model(frame, image_id):

    height, width = frame.shape[:2]
    image = cv2.resize(frame, (512, 512))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (512, 512), swapRB=True, crop=False)
    yolo.setInput(blob)
    outs = yolo.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    if (len(outs) > 0) :
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected

                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box

                    # Rectangle coordinates
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # confidence = 0.5 while threshold = 0.3
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    bbox_list = []

    if (len(boxes) > 0) :
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                data = { 
                    "category_id": int(class_ids[i]),
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    },
                    "score": confidences[i]
                }
                bbox_list.append(data)

        res = {
                "image_id": image_id,
                "bbox_list": bbox_list
            }
        return res
    
    else :
        return {
            "image_id": image_id,
            "bbox_list": []
        }


def run_image(frame, image_id):
    return run_model(frame, image_id)
    