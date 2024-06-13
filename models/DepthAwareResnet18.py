from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, _log_api_usage_once, conv1x1, conv3x3
import math
import torch
from torch import nn, Tensor
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
import collections.abc as container_abcs 
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from typing import Type, Any, Callable, Union, List, Optional



class ShapeConv2d(Module):
    """
       ShapeConv2d can be used as an alternative for torch.nn.Conv2d.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ShapeConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.testing = not self.training
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if M * N > 1:
            self.Shape = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            self.Base = Parameter(torch.Tensor(1))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)

            init_one = np.ones([1], dtype=np.float32)
            self.Shape.data = torch.from_numpy(init_zero)
            self.Base.data = torch.from_numpy(init_one)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([1, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(ShapeConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def compute_shape_w(self):
        # (input_channels, D_mul, M * N)
        Shape = self.Shape + self.D_diag  # (1, M * N, self.D_mul)
        Base = self.Base
        W = torch.reshape(self.weight, (self.out_channels // self.groups, self.in_channels, self.D_mul))
        W_base = torch.mean(W, [2], keepdims=True)  # (self.out_channels // self.groups, self.in_channels)
        W_shape = W - W_base  # (self.out_channels // self.groups, self.in_channels, self.D_mul)

        # einsum outputs (out_channels // groups, in_channels, M * N),
        # which is reshaped to
        # (out_channels, in_channels // groups, M, N)
        D_shape = torch.reshape(torch.einsum('ims,ois->oim', Shape, W_shape), self.weight.shape)
        D_base = torch.reshape(W_base * Base, (self.out_channels, self.in_channels // self.groups, 1, 1))
        DW = D_shape + D_base
        return DW

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        if M * N > 1 and not self.testing:  # train and val
            DW = self.compute_shape_w()
        else:   # test
            DW = self.weight

        return self._conv_forward(input, DW)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.testing = not self.training
        super(ShapeConv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                       missing_keys, unexpected_keys, error_msgs)
        if self.kernel_size[0] * self.kernel_size[1] > 1 and not self.training:
            self.weight.data = self.compute_shape_w()

import torch
import torch.nn as nn
import torch.nn.functional as F
from .aCNN import computeOffset
from torchvision.ops.deform_conv import DeformConv2d


from torchvision.models.resnet import ResNet, Bottleneck

class DepthAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DepthAwareConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, x):
        
        depth = x[:,-1:,:,:]
        x = x[:,:3,:,:]

        print("Shape of depth tensor:", depth.shape)
        print("Shape of x tensor:", x.shape)
        # Example of using depth to modify the convolutional operation
        depth_resized = F.interpolate(depth, x.size()[2:], mode='bilinear', align_corners=True)
        offset = computeOffset(depth_resized, self.conv.kernel_size[0], self.conv.stride[0])
        offset = F.pad(offset, (1, 1, 1, 1), "constant", 0)
        # x = x + offset  # This is a simple example; you can define a more complex interaction
        return self.conv(x)
    
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input
    
class PoolingModule(nn.Module):
    def __init__(self, inplanes, kernel_size=3, stride=1, padding=0, dilation=1):
        super(PoolingModule, self).__init__()
        self.pool = DeformConv2d(inplanes, inplanes, kernel_size=kernel_size, stride=stride,padding=padding,dilation=dilation, groups = inplanes)
        self.pool.weight.data.fill_(1/kernel_size**2)
        self.pool.bias.data.zero_()
        for param in self.pool.parameters():
            param.requires_grad = False 
    def forward(self, x, offset):
        x = self.pool(x, offset)
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)

class DepthResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # Change the line below in your code Aleksander
        self.conv1 = DepthAwareConv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _depth_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> DepthResNet:
    model = DepthResNet(block, layers, **kwargs)
    return model


def depthresnet18(**kwargs: Any) -> DepthResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _depth_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)