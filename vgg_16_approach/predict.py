#image path will have to provide like >> python3 predict.py -i test_image.jpg


#from numpy.core.fromnumeric import resize
import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())

input_image = args['input']


model = load_model(os.path.join(os.getcwd(),'outputs/detector_vgg16.h5'))
lb = pickle.loads(open(os.path.join(os.getcwd(),'outputs/lb.pickle'),'rb').read())


image = load_img(input_image, target_size=(224,224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image,axis=0)

bbx_pred, labels_pred = model.predict(image)
(sx,sy,ex,ey) = bbx_pred[0]


i= np.argmax(labels_pred, axis=1)
label = lb.classes_[i][0]
if label == 0 : 
    label = 'non-emergency'
else:
    label = 'emergency'



image = cv2.imread(input_image)
image = imutils.resize(image, width=600)
(h,w) = image.shape[:2]


sx = int(sx * w)
sy = int(sy * h)
ex = int(ex * w)
ey = int(ey * h)

y = sy - 10 if  sy -10 > 10 else sy +10
cv2.putText(image,label, (sx,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
cv2.rectangle(image, (sx,sy),(ex,ey), (0,255,0),2)

cv2.imshow('Output',image)
cv2.waitKey(0)

