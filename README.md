# Stereo Visual SLAM with YOLO-Based Object Detection

## Project Overview
This project integrates Stereo Visual SLAM with YOLO-based object detection
to improve mapping in dynamic environments.

##  Features
- ORB-SLAM trajectory estimation
- YOLO + MobileNet SSD object detection
- Ground Truth (GT) comparison
- ATE / RPE evaluation

##  Project Structure
- orb_slam_results/ → ORB-SLAM trajectory + results
- yolo_mobilenet_results/ → YOLO SLAM results
- code/ → implementation scripts

##  Dataset
- TUM RGB-D Dataset (not included in repo)

##  Results
- Improved trajectory accuracy using object detection
- Comparison plots included

##  How to Run
1. Run SLAM system
2. Run object detection
3. Compare with GT using evaluation script
