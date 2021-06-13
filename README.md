# emergency_vehicle_detection
## The main goal is to classify emergency vehicle(ambulance and fire fighter truck) and non-emergency vechicle(car,bus)  along with the bounding box

## The main database of kaggle's https://www.kaggle.com/abhisheksinghblr/emergency-vehicles-identification has some limitations. Because there were no bounding boxes

For finding the bounding box I with my friend, @Carbodose applied yolov4 pre-trained model that is applied in MS_COCO Dataset.. The darknet reporsitory was downloaded first then the huge image.c file needed to be modified. It was quite challenging.  


#This poroject is done followed by these awesome tutorials>>>
1.with VGG16 with
https://www.pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

2. with yolo
