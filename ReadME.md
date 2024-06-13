# Depth Adaptive Detection Transformer
[<img src="images/github-mark.svg" width="40">](https://github.com/DepthAwareObjectDetection/detr)
## Introduction
Stereo cameras are ubiquitous in autonomous vehicles. 
Can depth information from these cameras improve the detection performance of a state-of-the-art object detector? 
We investigate the use of depth aware convolution layers such as [ShapeConv](https://github.com/DepthAwareObjectDetection/ShapeConv) and [Depth Adaptive CNN](https://github.com/DepthAwareObjectDetection/Depth-Adapted-CNN) in the [DeTR](https://github.com/DepthAwareObjectDetection/detr) object detector.

We start of by introducing ShapeConv and Depth Adaptive CNNs first. Then we talk a bit about the Detection Transformer, it's backbone and how we modified it to used the depth convolution layers. In the end we shall discuss results and progression of training over time.

## ShapeConv
