#base path to yolo directory
MODEL_PATH = "yolo-coco"

#intialize minimum probability to filter weak detection along with the threshold when applying non-maxima supression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#boolean indicating if nvidia cuda gpu is used
USE_GPU = False

#define the minimum distance in pixel that the two people can be from each other
MIN_DISTANCE = 50 # in pilxel

