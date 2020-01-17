# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

import pickle
with open("data.dat", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
import random
random.shuffle(b)


d = list(b)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())


from PIL import Image
img = Image.open("/home/brucewayne/Downloads/tesseract-python/mid_frame.png")
img1=img.crop((0,0,35,35))
# load the example image and convert it to grayscale
cv2.imwrite("3.jpg",img1)
image=cv2.imread("1.jpg")
scale_percent = 60 # percent of original size
width = int(image.shape[1] * 2)
height = int(image.shape[0] * 2)
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
kernel = np.ones((5,5),np.uint8)
gray = cv2.dilate(gray,kernel,iterations = 1)
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)



text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
