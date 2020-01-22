import cv2
import numpy as np
import os

FILE_OUTPUT = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture('1.mp4')

currentFrame = 0
os.getcwd()


# Define the codec and create VideoWriter object
fourcc=cv2.VideoWriter_fourcc(*'mpeg')
out = cv2.VideoWriter('test.avi',fourcc, 60.0, (1920,1080))

# while(True):
try :
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Handles the mirroring of the current frame

            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame',frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
except :
    cap.release()
    out.release()
    cv2.destroyAllWindows()
