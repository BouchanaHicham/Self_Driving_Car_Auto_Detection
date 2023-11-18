import os

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)         # Pc Camera 0 for Phone
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')

# 0 Plastic
# 1 Crumbled Paper
# 2 Nothing
# 3 Glass
# 4 Electronics
# 5 Textiles
# 6 Metal

# Import The Bin Images
imgBinsList = []
pathFolderBins = "Bins_New"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

classDic = {0: 0,
            1: 1,
            }
Bin_Name_Dic = {
            0: "Person",
            1: "Truck",
}
bin_width = 200  # Adjust this to your preferred size
bin_height = 200  # Adjust this to your preferred size

while True:
    _, img = cap.read()
    prediction = classifier.getPrediction(img)
    print("prediction:",prediction)
    # --------------------------- [Predictions] -------------------------------------
    classID = prediction[1]
    probabilities = prediction[0] * 100  # Probability as a percentage

    formatted_probability = "{:.4f}%".format(
        probabilities[classID])  # Format probability as a percentage with 4 decimal places


    # formatted_probabilities = [f"{prob:.4f}%" for prob in probabilities]
    # print((f"All Probabilities: {formatted_probabilities}"))

    # Manually extract the coordinates of the bounding box for the object being classified
    x, y, w, h = (100, 100, 200, 200)  # You can replace these values with actual coordinates from your model

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the object


    print(f"Class ID: {classID}, Probability: {formatted_probability}")
    print("Bin:",Bin_Name_Dic[classID])
    #print("imgBinsList:",imgBinsList)
    # ----------------------------------------------------------------
#im not actually coding im just messing around for this guys to take pictures
    # ------------------------------------ [ Overlay (Bins) ]  ------------------------------------
    if classID != 2:
        bin_image = imgBinsList[classDic[classID]]

        # Ensure the bin image has 3 channels (remove alpha channel if present)
        if bin_image is not None and bin_image.shape[2] == 4:
            bin_image = bin_image[:, :, :3]

        if bin_image is not None:
            # Resize the bin image to the desired width and height
            bin_image = cv2.resize(bin_image, (bin_width, bin_height))

            # Define the position of the overlay in the bottom left corner
            overlay_x = 0
            overlay_y = img.shape[0] - bin_image.shape[0]  # Place it at the bottom

            # Place the bin image on the output image
            img[overlay_y:, overlay_x:overlay_x + bin_image.shape[1]] = bin_image
    # ----------------------------------------------------------------------------------
    cv2.imshow("Output", img)
    cv2.waitKey(1)