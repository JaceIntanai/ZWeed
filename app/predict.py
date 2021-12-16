# from picamera.array import PiRGBArray
# from picamera import PiCamera
import cv2, numpy as np, os
# from imutils.video import FPS
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True, help="path to the input video/image/ file or type webcam to open your webcam")
# ap.add_argument("-m", "--model", required=True, help="specify preferred model type: yolov4, \
#     yolov4-tiny, yolov4-csp, yolov4x-mish")
# args = vars(ap.parse_args())

# yolo = cv2.dnn.readNet(args["model"] + "_training_4000.weights", args["model"] + "_training.cfg")
current_dir = os.getcwd()
cfg = os.path.join("yolov4-tiny_training.cfg")
weights = os.path.join("yolov4-tiny_training_1000.weights")

yolo = cv2.dnn.readNetFromDarknet(cfg, weights)

classes = ["pipe", "corner", "flange", "anode"]
#classes = ["Defective", "Normal"]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
#font = cv2.FONT_HERSHEY_PLAIN

# fps = None

def run_model(frame, image_id):

    height, width = frame.shape[:2]

    # if width > 512 or height > 512:
    #     frame = cv2.resize(frame, (512, int(height*(512/width)))) if height > width \
    #         else cv2.resize(frame, (int(width*(512/height)), 512))
    image = cv2.resize(frame, (512, 512))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (512, 512), swapRB=True, crop=False)
    yolo.setInput(blob)
    outs = yolo.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

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

            # color = colors[class_ids[i]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(frame, classes[class_ids[i]], (x , y - 7), font, 0.5, color, 1)
    #cv2.imshow('frame', frame)
    res = {
            "image_id": image_id,
            "bbox_list": bbox_list
        }
    return res


def run_image(frame, image_id):
    # frame = np.array(cv2.imread(args["input"]))
    return run_model(frame, image_id)
    #cv2.waitKey(0)

#image = cv2.imread("/Users/pin/Documents/Hackathon/dataset_v1.0/images/20201107122805838.png")
#run_image(image, 1)

# if args["input"] == 'webcam':
    # initialize the camera and grab a reference to the raw camera capture
    # camera = PiCamera()
    # rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    # time.sleep(0.1)
    # vid = cv2.VideoCapture(0)

    # if not vid.isOpened():
    #     print("Cannot open camera")
    #     exit()

    # while vid.isOpened():
    #     ret, frame = vid.read()
    #     # image = frame.array
    #     run_model(frame)
    #     # clear the stream in preparation for the next frame
    #     # rawCapture.truncate(0)
    #     fps.stop()
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord("q"):
    #         break

# elif '.jpg' in args["input"] or '.png' in args["input"]:
#     run_image()
# else:
#     run_video()

# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))