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
		 write the grayscale image to disk as a temporary file so we can
		# apply OCR to it
		kernel = np.ones((5,5),np.uint8)
		gray = cv2.dilate(gray,kernel,iterations = 1)
		filename = "{}.png".format(os.getpid())
		cv2.imwrite(filename, gray)
		num=""
		num = int(pytesseract.image_to_string(Image.open(filename)))
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
q = 1
mv=20
run_val=[]
data = []
cap= cv2.VideoCapture('1.mp4')
store=[]
while cap.isOpened():
    ret,full = cap.read()
    import numpy as np
    if end is False :
        rects=[]
        # First Run
        rects=[]
        frame = full.copy()

        print(frame.shape)
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        import numpy as np
        kernel = np.ones((5,5))
        mask = cv2.inRange(frame,(170,170,170),(255,255,255))
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

        # Calculating Running Average
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
        if(mv >= 15 and len(rects) == 16):
            if len(data)>0  :
                thresh = 0.9
                for i in range(len(data)):
                    for k in range(len(rects)):
                        if(get_iou(data[i],rects[k]) > thresh):
                            for t in range(len(rects[i])):
                                data[i][t] = 0.5*(data[i][t] +rects[k][t])
                # initialize the bounding box coordinates of the object we are going
                # to track
                initBB = ( 8*data[q][0] , 8*data[q][1] , 8*(data[q][2]) , 8*(data[q][3]) )
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
            cv2.waitKey(1)
    if end :
        if Once:
            print(mid.shape)
            print(data)
            for i in range(len(data)):
                x,y,w,h = data[i]
                x,y,w,h = 8*int(x),8*int(y),8*int(w),8*int(h)
                cv2.rectangle(mid,(x,y),(x+w,y+h),(255,20,20),2)
            (x, y, w, h) = [int(v) for v in box]
            x_d,y_d , w_d,h_d =  8*data[q][0] , 8*data[q][1] , 8*(data[q][2]) , 8*(data[q][3])
            x_vec , y_vec = x - x_d , y - y_d
            for i in range(len(data)):
                data[i][0] = data[i][0]+x_vec/8
                data[i][1] = data[i][1]+y_vec/8
            Once = False
            roi = get_roi(mid,data)
            initBB = tuple(roi)
            tracker.init(full,initBB)
        (success, box) = tracker.update(full)
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(full, (x, y), (x + w, y + h),(0, 255, 0), 2)
        for i in range(len(data)):
            x,y,w,h = data[i]
            x,y,w,h = 8*int(x),8*int(y),8*int(w),8*int(h)
            cv2.rectangle(mid,(x,y),(x+w,y+h),(20,20,255),2)
        cv2.imshow("frame",mid)
        cv2.waitKey(0)
