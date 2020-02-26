import numpy as np
import cv2
import imutils
import time
from PIL import Image
import pytesseract
import os

cv2.namedWindow("cnts_canny_white",cv2.WINDOW_NORMAL)
cv2.namedWindow("cnts_canny_red",cv2.WINDOW_NORMAL)
cv2.namedWindow("red",cv2.WINDOW_NORMAL)
cv2.namedWindow("white",cv2.WINDOW_NORMAL)

def get_iou(x, y):
    boxA = list(x)
    boxB  = list(y)
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

cap= cv2.VideoCapture('./Train_videos/1.mp4')

x1,y1,w1,h1 = 0,0,0,0
threshold_iou = 0.95

conf_activate, max_val, tries, count = 0,0,0,0
is_moving, has_stopped, box_found = False, False, False
box = None

lower_white = np.array([0,0,194])
upper_white = np.array([178,37,255])
lower_red = np.array([90,79,71])
upper_red = np.array([178,225,255])

tracker = cv2.TrackerCSRT_create()

# Color masking and return contours
def color_masking(frame, lower_range, upper_range, sensitivity = 40,ret_cnts = True):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # (hMin = 0 , sMin = 0, vMin = 194), (hMax = 178 , sMax = 37, vMax = 255) white boxes

    mask = cv2.inRange(hsv, lower_range, upper_range)
    res1 = cv2.bitwise_and(frame, frame, mask=mask)
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
     # res1 = cv2.threshold(res1, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("white",res1)
    if not ret_cnts:
        return res1

    # gray_cnt = res1.copy()
    cnts = cv2.findContours(res1.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv2.drawContours(frame_copy2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("cnts_canny_white",frame_copy2)

    return res1,cnts

## Pytesseract ocr-implementation !to-change
def run_ocr(frame_ocr):
    filename = "{}.png".format(os.getpid())
    # cv2.imshow("Ocr_img",res1[y+10:y+h-10,x+10:x+w-10])
    cv2.imwrite(filename, frame_ocr)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

# Cleanup bad contours for red boxes
def cleanup_cnts(cnts):
    ## Add more conditions if needed
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))[-28:-3]
    cv2.drawContours(frame_copy1, cntsSorted, -1, (0, 255, 0), 3)
    area, coords = [], []

    ## Apply our conditions for deleting bad contours
    for cnt in cntsSorted:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(w/h)
        if 1 < w/h < 2:
            cv2.rectangle(frame_copy1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            area.append(w*h)
            coords.append([x, y, w, h])
    return zip(*sorted(zip(area, coords)))

while True:
    a = time.time()
    ret, frame = cap.read()
    frame_copy1 = frame.copy()
    frame_copy2 = frame.copy()

    #################################################
    ## When the box has not been found
    if not box_found and tries < 20:
        ## Return red boxes
        res, cnts = color_masking(frame, lower_red, upper_red)
        area, coords = cleanup_cnts(cnts)

        ## update iteration to activate ocr
        if len(area) >= 16:
            conf_activate += 1
        elif conf_activate > 3:
            conf_activate -= 1

        ## Only after conf_acivate frames of detecting more than 16 boxes do we start ocr
        ## Try to find the 500 box, n times , else best other box
        if conf_activate >= 15:
            gray_frame = color_masking(frame, lower_white, upper_white, ret_cnts=False)
            ## trigger ocr
            tries += 1
            for i in range(len(area)):
                x, y, w, h = coords[i]
                ocr_frame = gray_frame[y+10:y+h-10, x+10:x+w-10]
        
                text = run_ocr(ocr_frame)
                if text.isdigit() and int(text) >= max_val:
                    max_val = int(text)
                    x1, y1, w1, h1 = x, y, w, h

            print("Found {} containing box".format(max_val))
            cv2.rectangle(frame_copy1, (x1, y1), (x1+w1, y1+h1), (0, 0, 0), 14)
            if max_val >= 400 or tries == 20:
                tracker.init(frame, (x1, y1, w1, h1))
                box = (x1, y1, w1, h1)
                box_found = True
    #################################################

    #################################################
    ### The box has been found and is being tracked 
    ### till it stops
    if box_found and not has_stopped:
        old_box = box
        (success, box) = tracker.update(frame)
        (x, y, w, h) = [int(v) for v in box]

        iou_val = get_iou(box, old_box)
        if(iou_val < 0.90):
            is_moving = True
        if iou_val > threshold_iou:
            count += 1
        else:
            count = 0
        if count >= 20 and is_moving:
            print("\n\n\n\n Stopped")
            has_stopped = True
        cv2.rectangle(frame_copy1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #################################################

    #################################################
    ## The box has stopped moving output the number we
    ## need to select to scam :D
    if has_stopped:
        gray_frame, cnts = color_masking(frame, lower_white, upper_white)
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))[-28:-3]

        for cnt in cntsSorted:
            cv2.rectangle(frame_copy1, (x, y), (x + w, y + h), (0, 0, 0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            iou_val_1 = get_iou(box, (x, y, w, h))
            if iou_val_1 > 0.2:
                cv2.rectangle(frame_copy1, (x, y), (x + w, y + h), (255, 0, 255), 5)
                ocr_frame = gray_frame[y+10:y+h-10, x+10:x+w-10]
                text = run_ocr(ocr_frame)
                print("Select {}".format(text))
    #################################################
    cv2.imshow("cnts_canny_red", frame_copy1)
    print("fps is {}".format(-1/(a-time.time())))
    key = cv2.waitKey(0) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
# cleanup the cap and close any open windows
cap.release()
cv2.destroyAllWindows()