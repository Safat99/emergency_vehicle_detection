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

<!-- ## Methodology  

To generate bounding boxes, my collaborator **[@Carbodose](https://github.com/Carbodose)** and I applied a **YOLOv4 pre-trained model**, originally trained on the **MS COCO dataset**.  

We first downloaded the **Darknet** repository and modified parts of the `image.c` file to fit our use case — a process that was both technical and quite challenging. But because of this, we were able to **annotate thousands of images automatically** instead of doing manual labeling with tools.

The project also explored **VGG16-based classification** and **bounding box regression** to improve accuracy and robustness. -->

## Methodology  

To detect and classify emergency vehicles, different CNN-based approaches were tested — mainly using **VGG16** and **VGG19** architectures.  

### 1. Data Preparation  
The dataset was taken from Kaggle’s *Emergency Vehicles Identification* dataset. Since it did not have bounding box annotations, we generated them **automatically** using a **YOLOv4 pre-trained model** (which was originally trained on the MS COCO dataset).  

We first downloaded the **Darknet** repository and modified parts of the `image.c` file to fit our use case — a process that was both technical and quite challenging. But because of this, we were able to **annotate thousands of images automatically** instead of doing manual labeling with tools.  

After annotation, the dataset was split into **training (80%)** and **testing (20%)** sets. The bounding box coordinates and class labels were stored in CSV files for later use in model training.

### 2. Model Training and Architecture  

- **VGG16 Approach:**  
  A VGG16 base network (pre-trained on ImageNet) was used, followed by a custom fully-connected head. The model outputs both **class labels** and **bounding box coordinates**, trained together in a multi-output setup.  
  Initial experiments were run for **25 epochs**, and then extended to **250 epochs** for better performance.  
  The longer training led to improved accuracy and more stable bounding box predictions.

- **VGG19 Approach:**  
  A deeper **VGG19** network was used for comparison. This version included several **hyperparameter tuning experiments**, such as optimizing **batch size**, adding **layer normalization**, and adjusting **learning rate** and **optimizer parameters**.  
  The model was also trained for **250 epochs**, showing gradual improvements across epochs, especially in bounding box accuracy.

### 3. Implementation Notes  
Both models were implemented using **TensorFlow/Keras**. The training used a dual-loss setup:
- `class_label_loss` → for classification  
- `bounding_box_loss` → for bounding box regression  

The total loss was computed as a weighted combination of both. Accuracy and loss plots were generated for each epoch to monitor convergence.

---

## Results and Discussion  

Extensive experiments were conducted on both **VGG16** and **VGG19** architectures to analyze how different normalization techniques and training durations affect performance.  
The models were evaluated on two outputs: **class label accuracy** and **bounding box accuracy**.

---

### 🔹 VGG16 Results  

During the initial tuning (25 epochs), several configurations were tested:

| Setup | Epochs | Layer Normalization | Batch Normalization | Class Accuracy | BBox Accuracy | Observations |
|:------|:-------:|:-------------------:|:-------------------:|:---------------:|:--------------:|:--------------|
| 1 | 25 | ✅ | ✅ | 0.7957 | 0.8516 | Stable performance, normalization helped reduce loss. |
| 2 | 25 | ❌ | ✅ | **0.8022** | **0.8645** | Best accuracy at short training — removing layer norm improved generalization. |
| 3 | 25 | ❌ | ❌ | 0.8022 | 0.8580 | Slight drop in bounding box precision; model still consistent. |
| 4 | 100 | ❌ | ✅ | 0.7957 | 0.8516 | Accuracy plateaued after 100 epochs. |
| 5 | 250 | ❌ | ✅ | 0.8022 | 0.8215 | Overfitting began after long training; no major improvement. |

**Summary:**  
- The **best configuration** for VGG16 was with **batch normalization only** (no layer normalization).  
- Increasing epochs beyond 100 did not yield better results — in fact, **bounding box accuracy slightly decreased**, likely due to overfitting.  
- Optimal test results for VGG16 reached **~80.2% classification accuracy** and **~86.4% bounding box accuracy**.

---

### 🔹 VGG19 Results  

Similarly, the VGG19 model was trained under different configurations and epoch counts:

| Setup | Epochs | Batch Normalization | Class Accuracy | BBox Accuracy | Observations |
|:------|:-------:|:------------------:|:---------------:|:--------------:|:-------------|
| 1 | 25 | ❌ | 0.7591 | 0.8215 | Base performance; decent but less accurate than VGG16. |
| 2 | 25 | ✅ | 0.7376 | 0.8236 | Batch normalization slightly hurt classification, but improved stability. |
| 3 | 100 | ✅ | 0.7613 | 0.8193 | Gradual improvement, but still lower than VGG16. |
| 4 | 250 | ✅ | 0.7742 | **0.8280** | Longer training improved bounding box precision; classification moderately better. |

**Summary:**  
- The **VGG19 model** benefited from longer training (250 epochs), achieving **82.8% bounding box accuracy** and **77.4% classification accuracy**.  
- Although bounding box predictions were slightly more precise than VGG16, the model required significantly more computation and showed mild overfitting in later epochs.  
- Batch normalization stabilized training but didn’t substantially boost classification performance.

---

### 🔹 Comparative Insights  

| Metric | **VGG16 (Best)** | **VGG19 (Best)** | Notes |
|:--------|:----------------:|:----------------:|:------|
| Epochs | 25 | 250 | VGG16 converged faster. |
| Class Accuracy | **0.802** | 0.774 | VGG16 classified vehicles more reliably. |
| BBox Accuracy | **0.864** | 0.828 | VGG16’s shorter training achieved better bounding box results overall. |
| Training Stability | ✅ | ⚠️ | VGG16 was smoother, VGG19 needed tuning. |

**Overall Conclusion:**  
- **VGG16 outperformed VGG19** in both classification and bounding box accuracy when trained efficiently.  
- Removing **layer normalization** and keeping **batch normalization only** yielded the best results.  
- **VGG19**, while deeper, required longer training and more compute without major accuracy gains — though it demonstrated slightly better localization consistency.  

In general, the **VGG16 model offered the best trade-off between performance, training time, and model stability** for the emergency vehicle detection task.



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
