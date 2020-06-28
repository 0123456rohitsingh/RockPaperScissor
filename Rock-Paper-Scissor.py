# import the necessary packages
import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
import time
import cv2
import os
import time
from threading import Timer
import random
import multiprocessing
from multiprocessing import Process,Value


# construct the argument parse and parse the arguments
# Here instead of passing arguments we can also define static variable for confidence and threshold 
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
# load the class labels our YOLO model was trained on
labelsPath = os.path.join("yolo/yolo.names")
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
#COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
#	dtype="uint8")
COLORS = np.arange(9).reshape(3,3)
COLORS[0][0] = 0
COLORS[0][1] = 0
COLORS[0][2] = 255
COLORS[1][0] = 0
COLORS[1][1] = 255
COLORS[1][2] = 0
COLORS[2][0] = 255
COLORS[2][1] = 0
COLORS[2][2] = 0
print(COLORS)
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.join("yolo/yolov3_custom_train_4000.weights")
configPath = os.path.join("yolo/yolov3_custom_train.cfg")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
#vs = VideoStream(src=0).start()
time.sleep(2.0)
writer = None
(W, H) = (None, None)

def detect_RPS(frame2):
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        #blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(frame2, 1 / 255.0, (288, 288), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        #print(confidence)
                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        label="None"
        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame2, (x, y), (x + w, y + h), (255,0,0), 2)
                        text = "{}".format(LABELS[classIDs[i]])
                        label = text
                        #text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                         #       confidences[i])
                        #cv2.putText(frame2, text, (x, y - 5),
                         #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame2,len(idxs),label

cs=0
us=0
# loop over frames from the video file stream
while True:
        # read the next frame from the file
        (grabbed, frame2) = vs.read()
        
        #frame2 = vs.read()
        frame2 = imutils.resize(frame2, width=600)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
                (H, W) = frame2.shape[:2]
                
        frame1 = cv2.imread('intro.png')
        frame1 = cv2.resize(frame1, (600,450))
        
        cv2.putText(frame1, "Let's Play", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(frame1, "Computer : "+str(cs), (225, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame2, "You : "+str(us), (225, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        frame2,det,lab= detect_RPS(frame2)
        print(lab)
        if det>0:
                if lab!=plab:
                        icn = random.randint(0, 2)
                        
                if icn==0:
                        frame1 = cv2.imread('rock.png')
                        clab = 'Rock'
                elif icn==1:
                        frame1 = cv2.imread('paper.png')
                        clab = 'Paper'
                elif icn==2:
                        frame1 = cv2.imread('scissor.png')
                        clab = 'Scissor'
                frame1 = cv2.resize(frame1, (600,450))
                cv2.putText(frame1, "Computer : "+str(cs), (225, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(frame2, "You : "+str(us), (225, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                

                if lab!=plab:
                        if lab == "Rock" and clab == "Paper":
                                cs=cs+1
                                us=us
                                cv2.putText(frame1, "Computer Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                        elif lab == "Paper" and clab == "Rock":
                                cs=cs
                                us=us+1
                                cv2.putText(frame2, "You Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        elif lab == "Scissor" and clab == "Rock":
                                cs=cs+1
                                us=us
                                cv2.putText(frame1, "Computer Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                        elif lab == "Rock" and clab == "Scissor":
                                cs=cs
                                us=us+1
                                cv2.putText(frame2, "You Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        elif lab == "Paper" and clab == "Scissor":
                                cs=cs+1
                                us=us
                                cv2.putText(frame1, "Computer Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                        elif lab == "Scissor" and clab == "Paper":
                                cs=cs
                                us=us+1
                                cv2.putText(frame2, "You Won!!!", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        else:
                                cs=cs
                                us=us
                                cv2.putText(frame1, "Its a Tie...", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                                cv2.putText(frame2, "Its a Tie...", (225, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


        
        both = np.concatenate((frame1, frame2), axis=1)
        cv2.imshow('Rock-Paper-Scissor', both)
        plab=lab
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
                break

# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
vs.release()
vs.stream.release()
