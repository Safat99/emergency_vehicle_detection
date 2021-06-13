from sklearn.utils import shuffle
import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import os
import pandas as pd

print("[INFO} loading dataset...")
rows = open(config.annots_path).read().strip().split("\n")

data = [] #images
labels = []
bboxes = []
filenames = []
imagePaths = []


cars = pd.read_csv(os.path.join(config.images_path,'train_test_vgg_format.csv'))

for i in cars.IMAGE:
	image = load_img(os.path.join(config.images_path, str(i) + ".jpg"))
	image = img_to_array(image)
	
	data.append(image)
	imagePaths.append(os.path.join(config.images_path, str(i) + ".jpg"))

tmp = []

for i in range(len(cars)):
	for j in range(1,5):
		tmp.append(cars.iloc[i,j])

	labels.append(cars.CLASS[i])
	bboxes.append(tmp)
	tmp = []



# convert the data and labels to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype='float32')
imagePaths = np.array(imagePaths)


# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
train_images,test_images,trainLabels,testLabels,trainBboxs, testBboxs, trainPaths, testPaths = train_test_split(data, labels, bboxes, imagePaths, test_size=0.10,
	random_state=42, shuffle = True)

# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(config.test_path, "w")
f.write("\n".join('testPaths'))
f.close()



