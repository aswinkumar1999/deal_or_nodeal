from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    cv2.imshow("frame_detect",img)
    # cv2.waitKey(0)
    return img


class detector(object):

    def __init__(self,configPath,weightPath,metaPath):
        self.netMain = None
        self.metaMain = None
        self.altNames = None

        # global self.metaMain, netMain, self.altNames
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        
        # out = cv2.VideoWriter(
        #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        #     (darknet.network_width(netMain), darknet.network_height(netMain)))
        # print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                        darknet.network_height(self.netMain),3)
        


    def get_classification(self, frame_read):
        prev_time = time.time()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        rows = frame_read.shape[0]/frame_resized.shape[0]
        cols = frame_read.shape[1]/frame_resized.shape[1]
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        # print(detections)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # out.write(frame_resized)
        print(1/(time.time()-prev_time))
        # cv2.imshow('Demo', image)
        if cv2.waitKey(3) & 0xFF == 27 :
            pass

        scores = []
        classes = []
        boxes = []
        num_detections = len(detections)
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = dict(x = int(xmin*cols), y = int(ymin*rows))
            pt2 = dict(x = int(xmax*cols), y = int(ymax*rows))
            boxes.append({'topleft' : pt1, 'bottomright' : pt2})

            if detection[0].decode() == 'person': #is it person????
                classes.append('1')
            else:
                classes.append('0')
            
            scores.append(detection[1])
        # out.release()
        print(boxes, scores, classes, num_detections)
        return boxes, [scores], classes, num_detections
