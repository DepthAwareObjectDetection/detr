import torch
import torch.nn as nn
import torch.nn.functional as F
from .aCNN import computeOffset

from torchvision.models.resnet import ResNet, Bottleneck

class DepthAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DepthAwareConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, x, depth):
        # Example of using depth to modify the convolutional operation
        depth_resized = F.interpolate(depth[0], x.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth_resized, self.conv.kernel_size[0], self.conv.stride[0])
        offset = F.pad(offset, (1, 1, 1, 1), "constant", 0)
        x = x + offset  # This is a simple example; you can define a more complex interaction
        return self.conv(x)
    

class DepthAwareResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(DepthAwareResNet, self).__init__(block, layers, num_classes, zero_init_residual)
        
        # Replace the initial conv1 layer
        self.conv1 = DepthAwareConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace all convolutional layers in the blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                DepthAwareConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, depth_aware=True))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, depth_aware=True))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        depth = x[-1].unsqueeze(0)

        x = self.conv1(x, depth)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, depth)
        x = self.layer2(x, depth)
        x = self.layer3(x, depth)
        x = self.layer4(x, depth)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
class DepthAwareBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, depth_aware=False):
        super(DepthAwareBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        if depth_aware:
            self.conv1 = DepthAwareConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = DepthAwareConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = DepthAwareConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
    def forward(self, x, depth):
        identity = x
        depth = depth.unsqueeze(0)

        out = self.conv1(x, depth)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, depth)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, depth)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
