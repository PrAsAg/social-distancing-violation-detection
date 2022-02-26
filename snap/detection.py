#import the necessary package
from .social_distance_config import NMS_THRESH
from .social_distance_config import MIN_CONF
import numpy as np
import cv2 as cv

def detect_people(frame, net, ln, personIdx = 0):
    """Detects people from frame using yolo
    frame: Frame from video file or directly from webcam 
    net: initialized and trained yolo object detection model
    ln: Yolo cnn output layers names
    personIdx: The yolo model can detect many type of objects; this index is specially for the person class as we won't be considering any other
    objects
    """

    #grab the dimension of the frame and initialize the list of results
    (H, W) = frame.shape[:2]
    results = []
    #The results consist of 
    # (1) the person prediction probability, 
    # (2) bounding box coordinates for the detection, and 
    # (3) the centroid of the object.

    #construct a blob from input frame and then perform a forward pass of the yolo object detector, giving us our bounding boxes and associated
    #probability
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    #initialize our list of detected bounding boxes, centroids and confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    #loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            #extract the class id and confidence(i.e probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter the detection by (1) ensuring that the object detected was a person and (2) the minimum confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                #scale the bounding box coordinates back relative to the size of image, 
                #keeping in mind that yolo actually returns the center (x,y)- coordinates of the bounding box
                #followed by boxes' Width and Height
                box = detection[0:4] *np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                #use the center (x,y) coordinates to derive the top and left cornor of the bounding box
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                #update our list of bounding coordinates, centroid and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #apply non maxima supression to supress weak, overlapping bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH )

    #ensure atleast one detection exists
    if len(idxs) > 0:
        #loop over indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #update our result list to consistof the person predicting probability, bounding box coordinates and the centroid
            r = (confidences[i], (x,y, x+w, y+h), centroids[i])
            results.append(r)


    #return the list of results
    return results
