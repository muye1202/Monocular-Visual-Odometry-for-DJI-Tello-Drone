"""
Credit for Cansik's Yolo-hand-detection pretrained models!

https://github.com/cansik/yolo-hand-detection.git
"""


import djitellopy as tello
import cv2
import numpy as np
from yolo_hand_detector.yolo import YOLO


path = "/home/muyejia1202/ComputerVision/project/yolo_hand_detector/"
yolo_model = YOLO(path + "models/cross-hands-tiny-prn.cfg", path + "models/cross-hands-tiny-prn.weights", ["hand"])
yolo_model.size = 416
yolo_model.confidence = 0.5
hand_count = 1

def hand_detector(img, vis=True):
    """
    Output:
        cx: x location of hand's center
        cy: y location of hand's center
        confidence: how confident the detector is about the result
    """
    width, height, inference_time, results = yolo_model.inference(img)
    if len(results) > 0:
        id, name, confidence, x, y, w, h = results[0]
        cx = x + (w / 2)
        cy = y + (h / 2)

        if vis:
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            cv2.imshow("hand", img)
            cv2.waitKey(100)

        return cx, cy, confidence, width, height
    
    return -1, -1, -1, -1, -1


if __name__ == "__main__":
    drone = tello.Tello()
    drone.connect()
    print("Drone connected!")
    print("remaining battery: " + str(drone.get_battery()))
    drone.streamon()

    while True:
        img = drone.get_frame_read().frame
        img = cv2.resize(img, (480, 480))

        cx, cy, confidence, w, h = hand_detector(img)