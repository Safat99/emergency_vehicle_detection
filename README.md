<!-- # emergency_vehicle_detection
## The main goal is to classify emergency vehicle(ambulance and fire fighter truck) and non-emergency vechicle(car,bus)  along with the bounding box

## The main database of kaggle's https://www.kaggle.com/abhisheksinghblr/emergency-vehicles-identification has some limitations. Because there were no bounding boxes

For finding the bounding box I with my friend, @Carbodose applied yolov4 pre-trained model that is applied in MS_COCO Dataset.. The darknet reporsitory was downloaded first then the huge image.c file needed to be modified. It was quite challenging.  


#This poroject is done followed by these awesome tutorials that I have followed>>>
1.with VGG16 with
https://www.pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

2. This video will make you understand about the gist: https://www.youtube.com/watch?v=mjk4vDYOwq0

2. with yolo -->


# Emergency Vehicle Detection  

## Overview  
This project aims to **classify and detect emergency vehicles** — such as ambulances and fire trucks — as well as **non-emergency vehicles** like cars and buses.  
The model not only classifies the type of vehicle but also provides **bounding boxes** around detected objects in the image.

---

## Dataset  
The primary dataset used in this project is from Kaggle:  
[Emergency Vehicles Identification Dataset](https://www.kaggle.com/abhisheksingblr/emergency-vehicles-identification)  

However, the original dataset **did not include bounding box annotations**, which limited its usefulness for object detection tasks.

---

## Methodology  

To generate bounding boxes, my collaborator **[@Carbodose](https://github.com/Carbodose)** and I applied a **YOLOv4 pre-trained model**, originally trained on the **MS COCO dataset**.  

We first downloaded the **Darknet** repository and modified parts of the `image.c` file to fit our use case — a process that was both technical and quite challenging. But because of this, we were able to **annotate thousands of images automatically** instead of doing manual labeling with tools.

The project also explored **VGG16-based classification** and **bounding box regression** to improve accuracy and robustness.

---

## Tutorials & References  

This project was inspired and guided by several excellent tutorials and resources:

1. **Multi-Class Object Detection with Keras & TensorFlow (VGG16)**  
   [PyImageSearch Tutorial](https://www.pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)

2. **Overview Video**  
   [YouTube Explanation](https://www.youtube.com/watch?v=mjk4vDYOwq0)

3. **YOLOv4 Implementation**  
   Applied via the Darknet framework on the MS COCO dataset.

---

## Results & Future Work  
- The model successfully identifies and classifies emergency vs. non-emergency vehicles.  
- Future improvements could include:
  - Using **YOLOv8** or **Detectron2** for more accurate detection.  
  - Expanding the dataset with **manually labeled bounding boxes**.  
  - Deploying the model for **real-time detection** using video feeds.

---

## Contributors  
- [@Carbodose](https://github.com/Carbodose)  
- [@Safat99](https://github.com/Safat99)  
