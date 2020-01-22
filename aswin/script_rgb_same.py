import cv2
mv=20
val=[]
cap= cv2.VideoCapture('1.mp4')
while cap.isOpened():
    ret,frame = cap.read()

    rects=[]
    full = frame.copy()

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
    if(len(val)<5):
        val.append(len(rects))
    if(len(val)==5):
        val.pop(0)
        val.append(len(rects))
        mv=0
        for i in range(5):
            mv = mv+val[i]
        mv = mv/5
        print(mv)
    if (mv > 16 ):
        cv2.waitKey(1)
    if(len(rects)==16):
        cv2.waitKey(0)
    try :
        for i in range(16):
            x,y,w,h = rects[i]
            cv2.rectangle(full,(8*x,8*y),(8*(x+w),8*(y+h)),(20,20,255),2)
    except :
        pass
    cv2.imshow('frame',full)
    if(mv>10 and len(rects)<16):
        cv2.waitKey(0)
    cv2.waitKey(1)
