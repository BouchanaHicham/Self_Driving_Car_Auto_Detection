# Import the Open-CV extra functionalities
import cv2
import time
import threading

# This is to pull the information about what each object is called
classNames = []
classFile = "C:/Users/hicha/Downloads/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = "C:/Users/hicha/Downloads/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:/Users/hicha/Downloads/Object_Detection_Files/frozen_inference_graph.pb"

# This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below
    # print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

# Function for the timer
def countdown_timer(seconds,object):
    for remaining in range(seconds, 0, -1):
        print(f"Time remaining: {remaining} seconds")
        time.sleep(1)
    print("Time's up!")
    print(object + " crossed the road!")


# Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)

    # Specify the objects you want to consider
    # 1 -> This is for specified objects
    specified_objects = ["person","car"]

    # Below is the never ending loop that determines what will happen when an object is identified.
    Person_Crossed_The_Road = 0
    Car_Crossed_The_Road = 0
    while True:
        success, img = cap.read()
        # Below provides a huge amount of controll. the 0.45 number is the threshold number, the 0.2 number is the nms number)
        # result, objectInfo = getObjects(img,0.45,0.2)
        # 2 -> This is for specified objects
        result, objectInfo = getObjects(img, 0.45, 0.2,True,specified_objects)
        #print(objectInfo)
        if(objectInfo):
            Object = objectInfo[0][1]
            #print("Object Is: " + str(objectInfo[0][1]))
            if(Object == 'person' and Person_Crossed_The_Road == 0):
                print("Person detected! Waiting for 3 seconds...")
                timer_thread = threading.Thread(target=countdown_timer, args=(2,Object))
                timer_thread.start()

                Person_Crossed_The_Road = 1
            if (Object == 'car' and Car_Crossed_The_Road == 0):
                print("Car detected! Waiting for 2 seconds...")
                timer_thread = threading.Thread(target=countdown_timer, args=(2, Object))
                timer_thread.start()

                Car_Crossed_The_Road = 1

        cv2.imshow("Output", img)
        cv2.waitKey(1)


