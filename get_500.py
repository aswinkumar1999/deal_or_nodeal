import numpy as np
import cv2

import imutils
import time
from PIL import Image
import pytesseract
import os

cv2.namedWindow("cnts_canny_white",cv2.WINDOW_NORMAL)
cv2.namedWindow("cnts_canny_red",cv2.WINDOW_NORMAL)

cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
cv2.namedWindow("red",cv2.WINDOW_NORMAL)
cv2.namedWindow("white",cv2.WINDOW_NORMAL)

cv2.namedWindow("Ocr_img",cv2.WINDOW_NORMAL)



cap= cv2.VideoCapture('1.mp4')

conf_activate = 0
conf_deactivate = 0
max_val = 0
x1,y1,w1,h1 = 0,0,0,0

activated = False
tracker = cv2.TrackerCSRT_create()

while True:
    a = time.time()
    ret,frame = cap.read()
    frame_copy1 = frame.copy()
    frame_copy2 = frame.copy()
    cv2.imshow("Frame", frame)

    #################################################
    ## This part will return the white boxes

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # (hMin = 0 , sMin = 0, vMin = 194), (hMax = 178 , sMax = 37, vMax = 255) white boxes
    sensitivity = 40
    lower_white = np.array([0,0,194])
    upper_white = np.array([178,37,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res1 = cv2.bitwise_and(frame,frame, mask=mask)
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
    res1 = cv2.threshold(res1, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("white",res1)

    gray_cnt = res1.copy()
    cnts = cv2.findContours(gray_cnt.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv2.drawContours(frame_copy2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("cnts_canny_white",frame_copy2)
    #################################################

    #################################################
    ## After finding the 500 box, no need to find red boxes again, so stop

    if conf_deactivate > 20:
        pass
        print("\n\n Deactivated, no more red boxes \n\n")
    if conf_deactivate <= 20:
        ## This part will return the red boxes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # (hMin = 90 , sMin = 79, vMin = 71), (hMax = 179 , sMax = 225, vMax = 224) red boxes
        sensitivity = 40
        lower_red = np.array([90,79,71])
        upper_red = np.array([178,225,255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        cv2.imshow("red",res)

        ## Find and sort contours on area
        gray_cnt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray_cnt.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))[-28:-3]
        cv2.drawContours(frame_copy1, cntsSorted, -1, (0, 255, 0), 3)
        area = []
        coords = []

        ## Apply our conditions for deleting bad contours
        for cnt in cntsSorted:
            x,y,w,h = cv2.boundingRect(cnt)
            print(w/h)
            if(1<w/h<2):
                cv2.rectangle(frame_copy1, (x, y), (x+w, y+h), (255,0,0), 2)
                area.append(w*h)
                coords.append([x,y,w,h])

        area, coords = zip(*sorted(zip(area, coords)))
        print(len(area))


        if len(area)>=16:
            conf_activate += 1
        elif conf_activate > 3:
            conf_activate -= 1

        ## If fifteen frames with more than 16 red boxes activate our ocr
        if conf_activate >= 15:
            ## trigger ocr , tesseract
            activated = True
            print("\n\n Activated OCR \n\n")
            ## Try to find the 500 box, n times , else other box
            # res1 = cv2.bitwise_not(res1)
            for i in range(len(area)):
                # print(coords[i])
                x,y,w,h = coords[i]

                filename = "{}.png".format(os.getpid())
                cv2.imshow("Ocr_img",res1[y+10:y+h-10,x+10:x+w-10])
                cv2.imwrite(filename, res1[y+10:y+h-10,x+10:x+w-10])
                text = pytesseract.image_to_string(Image.open(filename))
                os.remove(filename)
                print(i, " -- ",text)
                if text.isdigit() and int(text)>= max_val:
                    max_val = int(text)
                    x1,y1,w1,h1 = x,y,w,h
                # key = cv2.waitKey(100) & 0xFF

            print("Found {} containing box".format(max_val))
            cv2.rectangle(frame_copy1, (x1, y1), (x1+w1, y1+h1), (0,0,0), 14)
            if max_val >= 400:
                tracker.init(frame,(x1,y1,w1,h1))
                conf_deactivate +=20

        if activated and len(area)<16:
            conf_deactivate += 1

    # cv2.rectangle(frame_copy1, (x1, y1), (x1+w1, y1+h1), (0,0,0), 14)
    #################################################
    ##### x1,y1,w1,h1 and frame  are the ones i have now , i need to track it ####
    (success, box) = tracker.update(frame)
    (x, y, w, h) = [int(v) for v in box]
    cv2.rectangle(frame_copy1, (x, y), (x + w, y + h),(0, 255, 0), 2)

    cv2.imshow("cnts_canny_red",frame_copy1)
    print("fps is {}".format(-1/(a-time.time())))
    key = cv2.waitKey(0) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the cap and close any open windows
cap.release()
cv2.destroyAllWindows()
