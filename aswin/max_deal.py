import numpy as np
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
from PIL import Image
import pytesseract
import os

FILE_OUTPUT = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
tracker_new = OPENCV_OBJECT_TRACKERS["csrt"]()

def get_roi(image,data):
    values_img =[]
    for i in range(len(data)):
        # Get Patch of the images
        x,y,w,h = data[i]
        x,y,w,h = 8*int(x),8*int(y),8*int(w),8*int(h)
        img = image[y:y+h, x:x+w]
        scale_percent = 80 # percent of original size
        width = int(img.shape[1] * 2)
        height = int(img.shape[0] * 2)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        kernel = np.ones((5,5),np.uint8)
        gray = cv2.dilate(gray,kernel,iterations = 1)
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        try :
            num = int(pytesseract.image_to_string(Image.open(filename)))
        except :
            num = 0
        os.remove(filename)
        values_img.append(num)
    print(values_img)
    values_img = np.array(values_img)
    id = np.argmax(values_img)
    return data[id]

def get_iou(x, y):
    boxA=4*[0]
    boxB=4*[0]
    boxA[0] = x[0]
    boxA[1] = x[1]
    boxA[2] = x[2]+x[0]
    boxA[3] = x[3]+x[1]

    boxB[0] = y[0]
    boxB[1] = y[1]
    boxB[2] = y[2]+y[0]
    boxB[3] = y[3]+y[1]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

end = False
st_now=0
Once=True

mv=20
run_val=[]
data = []
print("Here")
cap= cv2.VideoCapture('/home/lordgrim/deal_or_nodeal/Train_videos/1.mp4')
print("Here1")
store=[]
# fourcc=cv2.VideoWriter_fourcc(*'mpeg')
# out = cv2.VideoWriter("output.mp4", fourcc, 60.0, (1920,1080))
cv2.namedWindow("mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("boxes",cv2.WINDOW_NORMAL)
cv2.namedWindow("edged",cv2.WINDOW_NORMAL)
cv2.namedWindow("cnts_canny",cv2.WINDOW_NORMAL)
cv2.namedWindow("canny_Edged",cv2.WINDOW_NORMAL)


try :
    while cap.isOpened():
        ret,full = cap.read()

        frame_copy2 = full.copy()
        frame = cv2.GaussianBlur(full, (9, 9), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)
        cv2.imshow("canny_Edged",edged)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of white color in HSV
        # change it according to your need !
        sensitivity = 40
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow("white",res)


        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cv2.drawContours(frame_copy2, cnts, -1, (0, 255, 0), 3) 
        cv2.imshow("cnts_canny",edged)

        import numpy as np
        if end is False :
            rects=[]
            # First Run
            frame = full.copy()
            frame = cv2.pyrDown(frame)
            frame = cv2.pyrDown(frame)
            frame = cv2.pyrDown(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



            kernel = np.ones((5,5))
            mask = cv2.inRange(hsv,(0,0,0),(255,30,255))
            erode = cv2.erode(mask,kernel)
            dilate = cv2.dilate(erode,kernel)
            mask = dilate
            cv2.imshow("mask",mask)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
            cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))

            remove = []
            for i in range(len(cntsSorted)):
                x,y,w,h = cv2.boundingRect(cntsSorted[i])
                if((x+h/2 < 0.2*frame.shape[1]) or (x+h/2 > 0.8*frame.shape[1]) or (y+w/2 < 0.2*frame.shape[0]) or (y+w/2 > 0.8*frame.shape[0])):
                    remove.append(i)
            for i in range(1,len(remove)-1):
                del cntsSorted[remove[-i]]
            if(len(cntsSorted)>0 and len(remove)>0):
                del cntsSorted[remove[0]]
            # C_max = max(cnts,key = cv2.contourArea)
            for i in range(min(len(cntsSorted),16)):
                x,y,w,h = cv2.boundingRect(cntsSorted[-i])
                rects.append([x,y,w,h])

            # Cleaning :
            area = []
            for i in rects:
                area.append(i[2]*i[3])
            mean = np.mean(area)
            std = np.std(area)
            stats_z = [(s - mean)/std for s in area]
            rects = [d for (d, remove) in zip(rects, np.abs(stats_z) > 2) if not remove]


            ### Second Runs
            threshold = 0.85
            val = range(20,40)
            for j in val:
                rects_temp=[]
                frame = full.copy()
                frame = cv2.pyrDown(frame)
                frame = cv2.pyrDown(frame)
                frame = cv2.pyrDown(frame)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                kernel = np.ones((5,5))
                mask = cv2.inRange(hsv,(0,0,0),(255,j,255))
                erode = cv2.erode(mask,kernel)
                dilate = cv2.dilate(erode,kernel)
                mask = dilate

                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))

                remove = []
                for i in range(len(cntsSorted)):
                    x,y,w,h = cv2.boundingRect(cntsSorted[i])
                    if((x+h/2 < 0.2*frame.shape[1]) or (x+h/2 > 0.8*frame.shape[1]) or (y+w/2 < 0.2*frame.shape[0]) or (y+w/2 > 0.8*frame.shape[0])):
                        remove.append(i)
                for i in range(1,len(remove)-1):
                    del cntsSorted[remove[-i]]
                if(len(cntsSorted)>0 and len(remove)>0):
                    del cntsSorted[remove[0]]
                # C_max = max(cnts,key = cv2.contourArea)
                for i in range(min(len(cntsSorted),16)):
                    x,y,w,h = cv2.boundingRect(cntsSorted[-i])
                    rects_temp.append([x,y,w,h])
                # Cleaning :
                area = []
                for i in rects_temp:
                    area.append(i[2]*i[3])
                mean = np.mean(area)
                std = np.std(area)
                stats_z = [(s - mean)/std for s in area]
                rects_temp = [d for (d, remove) in zip(rects_temp, np.abs(stats_z) > 2) if not remove]
                left = rects_temp.copy()
                for i in range(len(rects)):
                    for k in range(len(rects_temp)):
                        if(get_iou(rects[i],rects_temp[k]) > threshold):
                            for t in range(len(rects[i])):
                                rects[i][t] = 0.5*(rects[i][t] +rects_temp[k][t])
                            try :
                                left.remove(rects_temp[k])
                            except :
                                pass
                while(len(rects)<16 and len(left) > 0):
                    rects.append(left.pop(0))
                # Cleaning :
                area = []
                for i in rects:
                    area.append(i[2]*i[3])
                mean = np.mean(area)
                std = np.std(area)
                stats_z = [(s - mean)/std for s in area]
                rects = [d for (d, remove) in zip(rects, np.abs(stats_z) > 2) if not remove]
            remove = []
            for i in range(len(rects)):
                x,y,w,h = rects[i]
                if((x+h/2 < 0.2*frame.shape[1]) or (x+h/2 > 0.8*frame.shape[1]) or (y+w/2 < 0.2*frame.shape[0]) or (y+w/2 > 0.8*frame.shape[0])):
                    remove.append(i)
            for i in range(1,len(remove)-1):
                del rects[remove[-i]]
            if(len(rects)>0 and len(remove)>0):
                del rects[remove[0]]
            # Calculating Running Average
            print("len :" + str(len(rects)))
            print(len(run_val))
            if(len(run_val)<5):
                run_val.append(len(rects))
            if(len(run_val)==5):
                run_val.pop(0)
                run_val.append(len(rects))
                mv=0
                for i in range(5):
                    mv = mv+run_val[i]
                mv = mv/5
                print(mv)
            if (mv > 16 ):
                cv2.waitKey(1)
            if(mv == 16 and len(rects) == 16):
                if len(data)>0  :
                    thresh = 0.9
                    for i in range(len(data)):
                        for k in range(len(rects)):
                            if(get_iou(data[i],rects[k]) > thresh):
                                for t in range(len(rects[i])):
                                    data[i][t] = 0.5*(data[i][t] +rects[k][t])
                    # initialize the bounding box coordinates of the object we are going
                    # to track
                    initBB = ( 8*(data[5][0]) , 8*(data[5][1]) , 8*(data[5][2]) , 8*(data[5][3]) )
                    print(initBB)
                    if initBB is not None:
                        tracker.init(full,initBB)
                else :
                    data = rects.copy()
            if ( mv < 10 ) :
                st_now = 1
            if(st_now):
                store.append(full)
            if  (mv > 10 and st_now == 1 ):
                st_now = 0
                mid = store[int(len(store)/2)]
                end = True
            (success, box) = tracker.update(full)
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(full, (x, y), (x + w, y + h),(0, 255, 0), 2)
            for i in range(len(rects)):
                x,y,w,h = rects[i]
                x,y,w,h = 8*int(x),8*int(y),8*int(w),8*int(h)
                cv2.rectangle(full,(x,y),(x+w,y+h),(20,20,255),2)
                cv2.imshow("boxes",full)
                # out.write(full)
                cv2.waitKey(1)
        if end :
            if Once:
                print(mid.shape)
                Once = False
                print(data)
                (x, y, w, h) = [int(v) for v in box]
                x_d,y_d , w_d,h_d =  8*data[5][0] , 8*data[5][1] , 8*(data[5][2]) , 8*(data[5][3])
                x_vec , y_vec = x - x_d , y - y_d
                for i in range(len(data)):
                    data[i][0] = data[i][0]+x_vec/8
                    data[i][1] = data[i][1]+y_vec/8
                roi = get_roi(mid,data)
                roi = [8*int(roi[0]),8*int(roi[1]),8*int(roi[2]),8*int(roi[3])]
                roi = tuple(roi)
                print(roi)
                tracker_new.init(full,roi)
            (success, box_b) = tracker_new.update(full)
            (x, y, w, h) = [int(v) for v in box_b]
            print(box_b)
            cv2.rectangle(full, (x, y), (x + w, y + h),(0, 255, 0), 2)
            cv2.imshow("boxes",full)
            # out.write(full)
            cv2.waitKey(1)
except :
    cv2.destroyAllWindows()
    # out.release()
    cap.release()
