from __future__ import print_function
import sys
import cv2
from random import randint
# from darkflow.net.build import TFNet
import tensorflow as tf
import numpy as np
import time
cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)

##########################################################
#finds iou between bb1 and bb2 
def get_iou(current_ybboxes, prev_ybboxes):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    # """
    # determine the coordinates of the intersection rectangle
    # print("Previous {}".format(prev_ybboxes))

    # print("Enter current yyboxes", current_ybboxes)
    current_ybboxes_arranged = [None]*len(prev_ybboxes)
    # max_match_current_yybox = [None]*len(prev_ybboxes)
    for i in range(len(prev_ybboxes)):
        bb2 = [prev_ybboxes[i][0],prev_ybboxes[i][1],prev_ybboxes[i][0]+prev_ybboxes[i][2],prev_ybboxes[i][1]+ prev_ybboxes[i][3]]
        max_iou = 0
        max_match_current_yybox = None
        for j in range(len(current_ybboxes)):
            bb1 = [current_ybboxes[j][0],current_ybboxes[j][1],current_ybboxes[j][0]+current_ybboxes[j][2],current_ybboxes[j][1]+ current_ybboxes[j][3]]
            x_left = max(bb1[0], bb2[0])
            y_top = max(bb1[1], bb2[1])
            x_right = min(bb1[2], bb2[2])
            y_bottom = min(bb1[3], bb2[3])
            if x_right < x_left or y_bottom < y_top:
                iou = 0.0
                continue
            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bb1_area = (bb1[2]-bb1[0])*(bb1[3] - bb1[1])
            bb2_area = (bb2[2]-bb2[0])*(bb2[3] - bb2[1])
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            # print("IOU BETWEEN ",bb1,bb2, iou)
            if iou > max_iou and iou > 0.5:
                max_iou = iou
                max_match_current_yybox = current_ybboxes[j]
        current_ybboxes_arranged[i] = max_match_current_yybox
                
    # print('Arranged {}'.format(current_ybboxes_arranged))

    return(current_ybboxes_arranged)

def usage():
    print("\n\nUsage is python trakor.py yolov2/yolov3/the_model")
    exit()

# if len(sys.argv) != 2:
#     usage()

# if sys.argv[1] == 'yolov2':
#     from yolov2 import detector
# elif sys.argv[1] == 'yolov3':
#     from yolov3 import detector
# elif sys.argv[1] == 'the_model':
#     from the_model import detector
# else:
#     usage()

class Tracker(object):
    def __init__(self,img,fps,detector):
        # if sys.argv[1] == 'yolov2':
        #     from yolov2 import detector
        # elif sys.argv[1] == 'yolov3':
        #     from yolov3 import detector
        # elif sys.argv[1] == 'the_model':
        #     from the_model import detector
        # else:
        #     usage()
        ## Here we initialize our model
        self.our_detector = detector
        ybboxes = []
        ## The model returns the boxes,scores,classes,num_boxes
        out = self.our_detector.get_classification(img)
        # Extract the detected bounding boxes and add them to ybboxes.
        num_detections = int(out[3])
        ## The model needs to be fed the image in this form
        img_np = img[:,:,[2,1,0]]
        # person_counter = 0
        for i in range(num_detections):
            classId = int(out[2][i])
            score = float(out[1][0][i])
            bbox = out[0][i]
            if score > 0.95:
                # print(score , end = '\r')
                x = int(bbox['topleft']['x'])
                y = int(bbox['topleft']['y'])
                right = int(bbox['bottomright']['x'])
                bottom = int(bbox['bottomright']['y'])
                img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
                bbox=(x,y,right-x,bottom-y)
                ybboxes.append(bbox)
                br = (right,bottom)
                tl = (x,y)
                cv2.rectangle(img,tl, br, (0,0,255), 2, 1)
        

        self.current_ybboxes = ybboxes
        self.prev_frame = img

        ## This is the confidence array of the confirmed,sure,detected people (is initialized with ones)
        ## When this drops less than 1, we say that a box/person has exited the frame {EXIT Case}
        self.frame_confidence = np.ones((len(self.current_ybboxes)))

        ## We create the tracker with the type that we will be using throughout
        self.tracker = cv2.TrackerCSRT_create()
        # tracker = cv2.TrackerMOSSE_create()


        ## These are the boxes/people that contain the unsure,maybe,detected people (does not contain sure people)
        self.new_detections_ybboxes = []

        ## This is the confidence array of unsure,maybe,detected people (is initialized with [0.3])
        ## When this confidence reaches 1, we say that the person has been detected/ensured long enough and we add him to 
        ##      our current_ybboxes 
        self.new_detections_confidence = np.array([])

        self.person_counter = len(self.current_ybboxes)

        self.confidence_sub = 0.06*30.0/fps # Default values of confidence to be sub/added if fps = 30 
        self.confidence_add = 0.1*30.0/fps
        print('fps = ', fps)

    def run_program(self,frame,fps):
        t1 = time.time()
        ## This array is where we will be storing all the confirmed,sure people detections of the precious frame
        prev_ybboxes = self.current_ybboxes
        prev_new_detections_ybboxes = self.new_detections_ybboxes
        # print("initual ", prev_ybboxes)
        # prev_sure_yyboxes = sure_ybboxes
        
        ## This array is where we will be storing all the confirmed arranged detections
        self.current_ybboxes = []
        
        ## We make a frame copy so that our algorithms wont be messed up by cv2.rect,etc-> their colors
        frame_copy=frame.copy()

        ## The model returns the boxes,scores,classes,num_boxes
        out = self.our_detector.get_classification(frame)
        
        print('time_detect = ',time.time() - t1)
        # Extract and visualize detected bounding boxes and them to current_ybboxes
        num_detections = int(out[3])
        ## The model needs to be fed the image in this form
        img_np = frame[:,:,[2,1,0]]
        for i in range(num_detections):
            classId = int(out[2][i])
            score = float(out[1][0][i])
            bbox = out[0][i]
            ## Class ID of 1, 3 is person, check others(later)
            ## Find good value of score (was 0.72) &*& tune
            if score > 0.65 and (classId == 1 or classId == 3):
                # print(score , end = '\r')
                x = int(bbox['topleft']['x'])
                y = int(bbox['topleft']['y'])
                right = int(bbox['bottomright']['x'])
                bottom = int(bbox['bottomright']['y'])
                img_np = np.ascontiguousarray(img_np, dtype=np.uint8)

                ## Add them
                bbox=(x,y,right-x,bottom-y) 
                self.current_ybboxes.append(bbox)

                ## Visualise
                br = (right,bottom)
                tl = (x,y)
                # frame_copy_all_detections = frame_copy.copy()
                cv2.rectangle(frame_copy, tl, br, (0,0,255), 2, 1)
        cv2.imshow("All detections", frame_copy)

        ## This contains all the detections by our model              
        all_detection_ybboxes = self.current_ybboxes

        ## These are arranged according to the previous ybboxes.
        ## The arrangement is done with the help of IoU.
        ## We map the current_ybboxes to their corresponding prev_ybboxes using iou and store in current_ybboxes_copy.
        current_ybboxes_copy = get_iou(self.current_ybboxes, prev_ybboxes)

        ## These are the boxes/people that contain the unsure,maybe,detected people (does not contain sure people)
        self.new_detections_ybboxes = [item for item in all_detection_ybboxes if item not in current_ybboxes_copy]

        print('Previous new ybboxes',prev_new_detections_ybboxes)
        print('New detections ',self.new_detections_ybboxes)
        
        ## These contain the people in unsure detections(new_detections) which we have observed for some frame and we have deemed them worthy to enter
        ## Basically their confidence has reached 1, so we add them to this list and at the end we add them to current_ybboxes
        confirmed_new_detections = []

        ## This (is_in_pre_new_detections) are the people that are unsure but have made an appearence in some recent frame, effectively their confidence
        ## hasn't reached 1 yet, they have possiblity to be sure poeple
        is_in_prev_new_detections_yyboxes = get_iou(self.new_detections_ybboxes,prev_new_detections_ybboxes)
        print('Is there in previous new and current new ',is_in_prev_new_detections_yyboxes)

        ## This (is_not_in_prev_new_detections) are the unsure people that have absolutely never appeared ever.
        ## This is the first time that they have beeen detected by our model.
        is_not_in_prev_new_detections_yyboxes = [item for item in self.new_detections_ybboxes if item not in is_in_prev_new_detections_yyboxes]
        print('Is not there in prev, but there in current new ybboxes ',is_not_in_prev_new_detections_yyboxes)

        ## Mmmmm... simply equated so we can do prev = new
        self.new_detections_ybboxes = is_in_prev_new_detections_yyboxes

        ## These contain indices to be deleted (either because they were a false positive or because they were confirmed)
        indices_to_delete = []

        for i in range(len(self.new_detections_ybboxes)):
            
            ## Basically if in our arranged new_detections is its None it means there was no box to represent corresponding prev box
            ##   in pre_new_detections
            if(self.new_detections_ybboxes[i]) is None:
                # print("\n\nCurrent is {} and prev is {} during flicker\n ".format(current_ybboxes[i],prev_ybboxes[i]))
                
                ## Then we initialize a tracker a give a chance to the prev_box, but we decrease the confidence.
                tracker = None
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(self.prev_frame, prev_new_detections_ybboxes[i])
                ok, bbox = tracker.update(frame)
        
                p1 = (round(bbox[0]), round(bbox[1]))
                p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
                cv2.rectangle(frame_copy, p1, p2, (255,255,255), 4, 1)

                ## Confidence change on flicker &*& tune
                self.new_detections_confidence[i] -= self.confidence_sub
                if self.new_detections_confidence[i] <= 0:
                    indices_to_delete.append(i)
                    # print("we have added ", i," to delete")
                    ##This is our exit case do appropriate stuff here areeeyyyyy
                    continue
                self.new_detections_ybboxes[i] = bbox
                # print("Current is {} and prev is {} during flicker\n ".format(current_ybboxes[i],prev_ybboxes[i]))
            else:
                p1 = (self.new_detections_ybboxes[i][0],self.new_detections_ybboxes[i][1])
                p2 = (self.new_detections_ybboxes[i][0] + self.new_detections_ybboxes[i][2],self.new_detections_ybboxes[i][1] + self.new_detections_ybboxes[i][3])
                cv2.rectangle(frame_copy, p1, p2, (0,0,0), 4, 1)

                ## Confidemce change on successful appear &*& tune
                self.new_detections_confidence[i] += self.confidence_add
                if self.new_detections_confidence[i] >=1:  

                    confirmed_new_detections.append(self.new_detections_ybboxes[i])
                    indices_to_delete.append(i)




        for i in sorted(indices_to_delete,reverse=True):
            self.new_detections_ybboxes.pop(i)
            prev_new_detections_ybboxes.pop(i)
            # frame_confidence.delete(i)
            self.new_detections_confidence = np.delete(self.new_detections_confidence,i)
            # print("hmmmmmm, meowwwww") 

        for i in range(len(is_not_in_prev_new_detections_yyboxes)):
            print(is_not_in_prev_new_detections_yyboxes)
            self.new_detections_ybboxes.append(is_not_in_prev_new_detections_yyboxes[i])
            self.new_detections_confidence = np.append(self.new_detections_confidence,[0.0])
        print(self.new_detections_confidence)
        print('New detection AFTER popp')
        print(self.new_detections_ybboxes)
        print("\n-------------------------")

                


        # print("---------\nPrev {} \n New {}\n".format(prev_ybboxes, current_ybboxes))    
        self.current_ybboxes = get_iou(self.current_ybboxes, prev_ybboxes)
        # print("\n \nPrev {} \n Arranged New {}\n".format(prev_ybboxes,current_ybboxes))
        ### Try to make a fn of f(IOU, no.fails) = confidence , try implement this later
        # using list comprehension + enumerate 
        # finding None indices in list  
        res = [i for i in range(len(self.current_ybboxes)) if self.current_ybboxes[i] == None]
        # print(res)
        # put,fix this later , sleepppsssss.........
        indices_to_delete = []
        for i in range(len(self.current_ybboxes)):
            if(self.current_ybboxes[i]) is None:
                # prnt("\n\nCurrent is {} and prev is {} during flicker\n ".format(current_ybboxes[i],prev_ybboxes[i]))
                tracker = None
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(self.prev_frame, prev_ybboxes[i])
                ok, bbox = tracker.update(frame)
        
                p1 = (round(bbox[0]), round(bbox[1]))
                p2 = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
                cv2.rectangle(frame_copy, p1, p2, (0,255,0), 2, 1)
                ## Confidence change on flicker &*& tune
                self.frame_confidence[i] -= self.confidence_sub/1.5
                if self.frame_confidence[i] <= 0:
                    indices_to_delete.append(i)
                    # print("we have added ", i," to delete")
                    ##This is our exit case do appropriate stuff here areeeyyyyy
                    continue
                self.current_ybboxes[i] = bbox
                # print("Current is {} and prev is {} during flicker\n ".format(current_ybboxes[i],prev_ybboxes[i]))
            else:
                p1 = (self.current_ybboxes[i][0],self.current_ybboxes[i][1])
                p2 = (self.current_ybboxes[i][0] + self.current_ybboxes[i][2],self.current_ybboxes[i][1] + self.current_ybboxes[i][3])
                cv2.rectangle(frame_copy, p1, p2, (255,0,0), 3, 1)
                #person_counter# Confidence change on not flicker &*& tune
                self.frame_confidence[i] += self.confidence_add

        for i in sorted(indices_to_delete,reverse=True):
            self.current_ybboxes.pop(i)
            prev_ybboxes.pop(i)
            # frame_confidence.delete(i)
            self.frame_confidence = np.delete(self.frame_confidence,i)
            # print("hmmmmmm, meowwwww")

        self.frame_confidence[self.frame_confidence > 1] = 1
        
        self.current_ybboxes = self.current_ybboxes + confirmed_new_detections
        self.person_counter += len(confirmed_new_detections) 
        self.frame_confidence = np.append(self.frame_confidence,np.ones(len(confirmed_new_detections)))
        # print("\n Prev {} \nFInal New {}\n".format(prev_ybboxes,current_ybboxes))
        # print(frame_confidence,"\n---------------------------------\n\n")
        # show frame
        cv2.imshow('MultiTracker', frame_copy)
        self.prev_frame = frame

        print("\n\n\n -------$$$$$$$$$$$$$$$$-----------------\nThe number of boiiiissss areeeeee  \n\n",self.person_counter,"\n -------$$$$$$$$$$$$$$$$-----------------")
        print('time = ',time.time() - t1)
        return self.person_counter
