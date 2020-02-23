import numpy as np
import cv2

def get_iou(x, y):

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



mv=20
run_val=[]
data = []
cap= cv2.VideoCapture('1.mp4')

while cap.isOpened():
    ret,full = cap.read()
    import numpy as np

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
        if data is not None :
            thresh = 0.9
            for i in range(len(data)):
                for k in range(len(rects)):
                    if(get_iou(data[i],rects[k]) > thresh):
                        for t in range(len(rects[i])):
                            data[i][t] = 0.5*(data[i][t] +rects[k][t])
        elif data is None :
            data.append(rects)

    for i in range(len(rects)):
        x,y,w,h = rects[i]
        x,y,w,h = 8*int(x),8*int(y),8*int(w),8*int(h)
        cv2.rectangle(full,(x,y),(x+w,y+h),(20,20,255),2)
    print (mv)
    cv2.imshow("boxes",full)
    cv2.waitKey(1)
