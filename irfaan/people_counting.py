import pdb
import cv2
import sys

def usage():
    print("Usage is python people_counting.py yolov2/yolov3/the_model img/camera")
    exit()

if len(sys.argv) != 3:
    usage()
our_detector = None

## Change paths to various models and weights/graphs
if sys.argv[1] == 'yolov2':
    from yolov2 import detector
    options = {
            'model': '/home/lordgrim/darkflow/cfg/yolo.cfg',
            'load': '/home/lordgrim/darkflow/bin/yolov2.weights',
            'threshold': 0.1,
            'gpu': 0.7 }
    our_detector = detector(options)

elif sys.argv[1] == 'yolov3':
    from yolov3 import detector
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    our_detector = detector(configPath, weightPath, metaPath)

elif sys.argv[1] == 'the_model':
    from the_model import detector
    PATH_TO_MODEL = "/home/lordgrim/Re-identification/frozen_inference_graph.pb"
    our_detector = detector(PATH_TO_MODEL)


else:
    usage()

from tracker_class import Tracker
cap = cv2.VideoCapture("/home/lordgrim/Re-identification/sports_complex.mp4")
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
our_tracker = Tracker(frame,fps,our_detector)
frame_number = 1
print("Frame number is",frame_number)

while True:
    ret, frame = cap.read()
    people_count = our_tracker.run_program(frame,fps)
    cv2.imshow("frame",frame)
    frame_number += 1
    print("Frame number is",frame_number)
    if cv2.waitKey(0) & 0xFF == 27:  # Esc pressed
        break