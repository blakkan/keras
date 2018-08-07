# import the necessary packages
from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
ap.add_argument("-i", "--image", required=True,
	help="path to image file to classify")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())

# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["cat", "dog"]

# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])

# load the image, resize it to a fixed 32 x 32 pixels (ignoring
# aspect ratio), and then extract features from it
print("[INFO] classifying %s", args["image"] )

image = cv2.imread(args["image"])
features = image_to_feature_vector(image) / 255.0
features = np.array([features])

# classify the image using our extracted features and pre-trained
# neural network
probs = model.predict(features)[0]
prediction = probs.argmax(axis=0)

# draw the class and probability on the test image and display it
# to our screen
label = "{}: {:.2f}%".format(CLASSES[prediction],
		probs[prediction] * 100)
print( label )
