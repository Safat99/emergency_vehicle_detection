# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:11:48 2020

@author: Shamim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
import cv2

##########################           PART ONE            ##########################

yolo = pd.read_csv("beforeYOLO_MaxOnly_train+test.csv")

dataSize=yolo.shape[0]
dataSize = 5
filename = str(yolo.iloc[0,5])
newFname = filename

for i in range(len(yolo)):
	
	filename = str(yolo.iloc[i,5])
	newFname = filename
	# Load an color image in grayscale
	img = cv2.imread("cars/"+filename+".jpg",1)
	if(yolo.iloc[i,0]):
		#Emergency
		cv2.imwrite("Emergency/"+filename+".jpg",img)
	else:
		#Not Emergency
		cv2.imwrite("Not Emergency/"+filename+".jpg",img)



