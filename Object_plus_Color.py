# Importing necessary modules
import cv2
import numpy as np
import time
import threading

# Specify the paths for object detection
classFile = "C:/Users/hicha/Downloads/Object_Detection_Files/coco.names"
configPath = "C:/Users/hicha/Downloads/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:/Users/hicha/Downloads/Object_Detection_Files/frozen_inference_graph.pb"

# Load object detection model
classNames = []
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to get detected objects
def get_objects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    object_info = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                object_info.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, object_info

# Function for the timer
def countdown_timer(seconds, object):
    for remaining in range(seconds, 0, -1):
        print(f"Time remaining: {remaining} seconds")
        time.sleep(1)
    print("Time's up!")
    print(object + " crossed the road!")

# Function to execute when it is Green
def charge():
    print("Charging...")

# Function to execute when it is Red
def discharge():
    print("Discharging...")

# Function to execute when it is Blue
def wait():
    print("Waiting...")

# Capturing webcam footage
webcam_video = cv2.VideoCapture(0)
webcam_video.set(3, 640)
webcam_video.set(4, 480)

# Specify the objects you want to consider
specified_objects = ["person", "car"]

# Below is the never-ending loop that determines what will happen when an object is identified.
person_crossed_the_road = 0
car_crossed_the_road = 0

while True:
    success, img = webcam_video.read()
    
    # Object detection
    result, object_info = get_objects(img, 0.45, 0.2, True, specified_objects)

    if object_info:
        detected_object = object_info[0][1]

        if detected_object == 'person' and person_crossed_the_road == 0:
            print("Person detected! Waiting for 3 seconds...")
            timer_thread = threading.Thread(target=countdown_timer, args=(3, detected_object))
            timer_thread.start()
            person_crossed_the_road = 1

        if detected_object == 'car' and car_crossed_the_road == 0:
            print("Car detected! Waiting for 2 seconds...")
            timer_thread = threading.Thread(target=countdown_timer, args=(2, detected_object))
            timer_thread.start()
            car_crossed_the_road = 1

    # Color detection
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask_red = cv2.inRange(img_hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_green = cv2.inRange(img_hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_blue = cv2.inRange(img_hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_red) != 0:
        for contour in contours_red:
            if cv2.contourArea(contour) > 500:
                discharge()
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(img, "Red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if len(contours_green) != 0:
        for contour in contours_green:
            if cv2.contourArea(contour) > 500:
                charge()
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, "Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if len(contours_blue) != 0:
        for contour in contours_blue:
            if cv2.contourArea(contour) > 500:
                wait()
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(img, "Blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
