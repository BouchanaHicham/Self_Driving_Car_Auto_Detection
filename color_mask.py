# Importing all modules
import cv2
import numpy as np

# Specifying upper and lower ranges of color to detect in hsv format
lower_red = np.array([160, 50, 50])
upper_red = np.array([180, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Function to execute when it is Green
def Charge():
    print("Charging...")

# Function to execute when it is Red
def Discharge():
    print("Discharging...")

# Function to execute when it is Blue
def Wait():
    print("Waiting...")

# Capturing webcam footage
webcam_video = cv2.VideoCapture(0)

while True:
    success, video = webcam_video.read()  # Reading webcam footage
    
    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    mask_red = cv2.inRange(img, lower_red, upper_red)  # Masking the image to find the Red color
    mask_green = cv2.inRange(img, lower_green, upper_green)  # Masking the image to find the Green color
    mask_blue = cv2.inRange(img, lower_blue, upper_blue)  # Masking the image to find the Blue color

    mask_contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in the Red mask
    mask_contours_green, hierarchy_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in the Green mask
    mask_contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in the Blue mask

    # Finding the position of all contours for Red
    if len(mask_contours_red) != 0:
        
        for mask_contour in mask_contours_red:
            if cv2.contourArea(mask_contour) > 500:
                Discharge()
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Drawing rectangle for Red
                cv2.putText(video, "Red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text for Red

    # Finding the position of all contours for Green
    if len(mask_contours_green) != 0:
        
        for mask_contour in mask_contours_green:
            if cv2.contourArea(mask_contour) > 500:
                Charge()
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Drawing rectangle for Green
                cv2.putText(video, "Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text for Green

    # Finding the position of all contours for Blue
    if len(mask_contours_blue) != 0:
        
        for mask_contour in mask_contours_blue:
            if cv2.contourArea(mask_contour) > 500:
                Wait()
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Drawing rectangle for Blue
                cv2.putText(video, "Blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Adding text for Blue

    cv2.imshow("mask image", cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green), mask_blue))  # Displaying the combined mask image
    cv2.imshow("window", video)  # Displaying the webcam image

    cv2.waitKey(1)
