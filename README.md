# YOLOv7-Pose for Keypoint Detection

## Overview
This repository provides a guide for implementing keypoint detection using the YOLOv7-Pose model. Keypoint detection can be applied to various domains like pose estimation, object tracking, and activity recognition.

Refer to this [video tutorial](https://www.youtube.com/watch?v=OMJRcjnMMok&t=1s) for a detailed walkthrough.

---

## Steps to Get Started

### Step 1: Data Preparation using Roboflow
1. **Create a New Project:** Start by creating a new project in [Roboflow](https://roboflow.com/).
2. **Upload Data:** Upload your dataset to Roboflow.
3. **Annotate Images:** Use Roboflow’s annotation tool to label keypoints in your images.
4. **Review Annotations:** Verify and adjust annotations to ensure accuracy.
5. **Generate Dataset:** Create the final dataset in the required YOLOv7 format (YOLOv5 format can be used if YOLOv7 is unavailable).
6. **Download Dataset:** Save the dataset locally.

---

### Step 2: Clone the Repository
Clone this GitHub repository to your local machine:
```bash
git clone https://github.com/AarohiSingla/YOLOv7-POSE-on-Custom-Dataset.git
```

---

### Step 3: Clone the Repository
Install the required libraries/packages.
```bash
pip install -r requirements.txt
```

---

### Step 4: Place Your Dataset
Place your prepared dataset into the 'dataset' folder within the cloned repository.
The 'dataset' folder has a blueprint of the expected .yaml file for reference.

---

### Step 5: Download the required .pt model
Download the 'yolov7-w6-pose.pt' from the link [GitHub](https://github.com/WongKinYiu/yolov7/releases).

---

### Step 6: Train YOLOv7 Model for Keypoint Detection
Run the following command to start training:
```bash
python train.py --data data/custom_kpts.yaml \
                --cfg cfg/yolov7-w6-pose_custom.yaml \
                --hyp data/hyp.pose.yaml \
                --device 0 \
                --kpt-label \
                --epochs 600
```
- **Arguments:**
  - `--data`: Path to the dataset configuration file.
  - `--cfg`: Path to the YOLOv7 model configuration file.
  - `--hyp`: Path to the hyperparameter configuration file.
  - `--device`: Specify the GPU/CPU for training.
  - `--kpt-label`: Enable keypoint labeling.
  - `--epochs`: Number of training epochs.

---

### Step 7: Testing and Keypoint Detection
After training, use the following command to test the model and perform keypoint detection:
```bash
python detect.py --weights runs/train/exp3/weights/best.pt \
                 --kpt-label \
                 --source 1.jpg \
                 --conf 0.030 \
                 --iou 0.30
```
- **Arguments:**
  - `--weights`: Path to the trained model weights.
  - `--kpt-label`: Enable keypoint detection.
  - `--source`: Path to the input image or video.
  - `--conf`: Confidence threshold for detection.
  - `--iou`: Intersection over Union (IoU) threshold for detection.

---

## Notes
- Ensure all dependencies are installed before proceeding with training or testing.
- Use a GPU for faster training and inference.
- Adjust hyperparameters and dataset configurations for better performance based on your specific use case.

---

Happy coding!
