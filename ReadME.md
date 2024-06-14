# DA-DETR: Depth Adaptive Detection Transformer
[<img src="images/github-mark-white.svg" width="40">](https://github.com/DepthAwareObjectDetection/detr)
## Introduction
*Can depth information from stereo cameras improve the detection performance of a state-of-the-art object detector?*

Lately, depth aware convolution layers like the [ShapeConv](https://github.com/DepthAwareObjectDetection/ShapeConv) and [Depth Adaptive CNN](https://github.com/DepthAwareObjectDetection/Depth-Adapted-CNN) have shown great promise in improving performance on RGB-D datasets. 
We research the impact of using these depth/ shape aware convolution layers (in place of vanilla convolutions) in the [DeTR](https://github.com/DepthAwareObjectDetection/detr) object detector. 

The rest of the blogpost is organised as follows: We start of by introducing ShapeConv and Depth Adaptive CNNs first. 
Then we talk about the Detection Transformer, it's backbone and how we modify it to use depth convolution layers. 
Finally, we discuss results and progression of training over time.

## DeTR
<img src="images/DETR.png" width="">

**DeTR** offers an object detection pipeline combining an **CNN** with a **Transformer** architecture.
The original model matches Faster R-CNN with a ResNet-50, obtaining 42 AP on COCO using half the computation power (FLOPs) and the same number of parameters.

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. 
It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image 
context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

**The Backbone** of DeTR consists of the CNN creating the set of image features.

*For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.*

## ShapeConv
Shape-aware Convolutional layer (ShapeConv) processes depth information by decomposing it into different components.
Key points are as follows
- ShapeConv decomposes the input image into a shape component and a base component
- The shape components inform where the image is
- The base component informs what the image is
- ShapeConv leverages two trainable set of weights for each component as compared to one 'weight' for the vanilla convolutional layer
<img src="images/shape_conv.png" width="">
*A high level overview of ShapeCong*

## Depth Adaptive CNN


## View-of-Delft dataset
The View-of-Delft (VoD) dataset is a novel automotive dataset containing 8600 frames of synchronized and calibrated 64-layer LiDAR-, (stereo) camera-, and 3+1D radar-data acquired in complex, urban traffic. It consists of more than 123000 3D bounding box annotations, including more than 26000 pedestrian, 10000 cyclist and 26000 car labels.
<img src="images/labels.gif" width="">

## Changing the architecture
### Getting depth information for the RGB camera
The VOD dataset containes synchronised data acquired from cameras and a LiDAR. 
We get the depth information for the camera image by following these steps:

- We first transforming the LiDAR point cloud to the camera's frame of reference.
- We then use the camera's projection matrix to get the location of these transformed point clouds in the image frame. This representation is quite sparse
- To get a more dense representation, we use the depth information of neighboring pixels and calculated a weighted average. 
- Link to the original repository: https://github.com/BerensRWU/DenseMap
<!-- @Matthijs please add images here -->
### Using 4d data in DETR
- The DeTR uses *torchvision.datasets.CocoDetection* class to get the dataset
- This class uses the PIL library to read RGB images
- We overode the *_load_image* class method to make it read numpy arrays (4-channel) and convert them to an object of PIL.Image
### Adapting DeTR to use ShapeConv Convolution layer
- PyTorch provides a modular implementation of the ResNet as part of its torchvision library
- We modified the existing ResNet class definition to replace the first convolution layer (kernel size = 7) with the ShapeConv convolution layer
- The authors of ShapeConv provide ShapeConv2D -> A torch.nn module that is a drop in replacement for the vanilla Conv2D module.
- This new class is called DepthResNet (DRN).
- We use the DepthResNet-18 (Depth equivalent of ResNet18) instead of DRN-50/ 100 used in the DeTR traditionally. This is done to make it trainable on a laptop GPU.

### Adapting DeTR to use Depth adaptive Convolution layer

#### Freezing the transformer


## Results

### Fine tuning


### DeTR from scratch


### ShapeConv from scratch


### Comparison


## Discussion



## Future improvements




