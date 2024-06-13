# Depth Adaptive Detection Transformer
[<img src="images/github-mark.svg" width="40">](https://github.com/DepthAwareObjectDetection/detr)
## Introduction
Can depth information from these cameras improve the detection performance of a state-of-the-art object detector? 
Depth aware convolution layers like the [ShapeConv](https://github.com/DepthAwareObjectDetection/ShapeConv) and [Depth Adaptive CNN](https://github.com/DepthAwareObjectDetection/Depth-Adapted-CNN) have shown great promise in improving performance on RGB-D datasets. 
We research the impact of using these depth/ shape aware convolution layers (in place of vanilla convolutions) in the [DeTR](https://github.com/DepthAwareObjectDetection/detr) object detector. 

We use the [view-of-delft](https://github.com/tudelft-iv/view-of-delft-dataset) dataset to compare the models. Within this dataset, we use the RGB camera image along with the transformed LiDAR points for depth information. 

The rest of the blogpost is organised as follows: We start of by introducing ShapeConv and Depth Adaptive CNNs first. Then we talk about the Detection Transformer, it's backbone and how we modify it to use depth convolution layers. Finally, we discuss results and progression of training over time.


## ShapeConv
Shape-aware Convolutional layer (ShapeConv) processes depth information by decomposing it into different components -> a shape component and a base component.

$\sqrt{3x-1}+(1+x)^2$