#importing the necessary packages
from snap import social_distance_config as config
from snap.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2 as cv
import os


#constuct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help= "path to (optional) input video file")
ap.add_argument("-o", "--output", type= str, default= "", help = "path to (optional) output video file")
ap.add_argument("-d", "--display", type= int, default=1, help= "wether or not output frame should be displayed")
args = vars(ap.parse_args())


# --input: The path to the optional video file. If no video file path is provided, your computer’s first webcam will be used by default.
# --output: The optional path to an output (i.e., processed) video file. If this argument is not provided, the processed video will not be 
# exported to disk.
# --display: By default, we’ll display our social distance application on-screen as we process each frame. Alternatively, you can set this value 
# to 0 to process the stream in the background.


#load the coco class labels our model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#derive the paths to YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

#load our YOLO object model trained on coco dataset
print("[Info] loading yolo from disk...........")
net= cv.dnn.readNetFromDarknet(configPath, weightsPath)

#check if we are going to use GPU
if config.USE_GPU :
    #set CUDA as the preferable backend and target
    print("[Info] setting preferable backend and target to CUDA............")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

#Determining only the "output" layer names that we need from yolo
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

#Initialize the video stream and pointer to the output video file
print("[Info] accessing video stream.......")
vs = cv.VideoCapture(args["input"] if args["input"] else 0)
writer = None

#loop over the frames from the video stream
while True:
    #read the next frame from the file
    (grabbed, frame) = vs.read()
    #if the frame was not grabbed than we have reached the end of the stream
    if not grabbed:
        break

    #resize the frame and then detect people in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    #initialize the set of indexes that violated minimum social distance
    violate = set()

    #ensure there are atleast two people detections (required inorder to compute our pairwise distance maps)
    if len(results) >= 2:
        #extract all centroids from the results and compute the Euclidean distance between all pair of centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric= "euclidean")

        #loop over the upper triangular of the distance metric
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                #check to see if the distance between any two centroid pairs is less than the configured number of pixels
                if D[i,j] < config.MIN_DISTANCE:
                    #update our violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    #visualizing
    #loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then initialize the color of annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        #if the index pair exists in violation set then update the color
        if i in violate:
            color = (0, 0, 255)

        #draw the bounding box around the person and centroid coordinates of the person
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv.circle(frame, (cX, cY), 5, color, 1)


    #Draw the total number of social distance violation on the output frame
    text = "Social Distance Violations: {}".format(len(violate))
    cv.putText(frame, text, (10, frame.shape[0] - 25), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 3)

    #Check to see if our output frame should be displayed to our screen
    if args["display"] > 0:
        #show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF

        #if the 'q' key was pressed break from the loop
        if key == ord("q"):
            break

    # if the output video file path has been supplied and the video writer hasnot been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True) 

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(frame)


