import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, ch=64):
        self.inplanes = ch
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ch, layers[0])
        self.layer2 = self._make_layer(block, ch * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, ch * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, ch * 8, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        features.insert(0, x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        features.insert(0, x)
        x = self.layer3(x)

        features.insert(0, x)
        x = self.layer4(x)

        features.insert(0, x)

        return x, features


def resnet18(ch=64, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], ch=ch, **kwargs)

    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(ch=64, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], ch=ch, **kwargs)
    return model


def resnet101(ch=64, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], ch=ch, **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class ResEncodeUNet(nn.Module):
    """
    """

    def __init__(self, out_channels, final_sigmoid, interpolate=True, init_channel_number=64, resname="50"):
        super(ResEncodeUNet, self).__init__()

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        if resname=="50":
            self.encoders = resnet50(ch=int(init_channel_number/4))
        elif resname=="101":
            self.encoders = resnet101(ch=int(init_channel_number / 4))

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate),
            Decoder(int(init_channel_number/4) + 2 * init_channel_number, init_channel_number, interpolate)
        ])

        # in the last layer a 1×1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # encoder part
        x, encoders_features = self.encoders(x)

        encoders_features = encoders_features[1:]
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


class DoubleConv(nn.Sequential):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crb'):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
            # self._add_conv(1, in_channels, out_channels, kernel_size, order)

            # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size, order)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size, order)
        # self._add_conv(3, conv2_out_channels, conv2_out_channels, kernel_size, order)

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order):
        """Add the conv layer with non-linearity and optional batchnorm

        Args:
            pos (int): the order (position) of the layer. MUST be 1 or 2
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """
        assert pos in [1, 2, 3], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r' or 's' or 'l' in order, "'r' (ReLU layer) MUST be present"
        assert order[0] is not 'r' or 'l', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('relu{}'.format(pos), nn.ReLU(inplace=True))
            elif char == 'l':
                self.add_module('leakyrelu{}'.format(pos), nn.LeakyReLU(inplace=True))
            elif char == 's':
                self.add_module('sigmoid{}'.format(pos), nn.Sigmoid())
            elif char == 'c':
                self.add_module('conv{}'.format(pos), nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm2d MUST go after the Conv2d'
                self.add_module('norm{}'.format(pos), nn.GroupNorm(1, out_channels))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    self.add_module('norm{}'.format(pos), nn.BatchNorm2d(in_channels))
                else:
                    self.add_module('norm{}'.format(pos), nn.BatchNorm2d(out_channels))
            else:
                raise ValueError(
                    "Unsupported layer type '{}'. MUST be one of 'b', 'r', 'c'".format(char))


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        is_max_pool (bool): if True use MaxPool2d before DoubleConv
        max_pool_kernel_size (tuple): the size of the window to take a max over
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, is_max_pool=True, order='crg'):
        super(Encoder, self).__init__()
        self.max_pool_size = 3
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=3, order=order)

    def forward(self, x):
        if self.max_pool is not None:
            X_diff = x.shape[2] % self.max_pool_size
            Y_diff = x.shape[3] % self.max_pool_size
            if X_diff + Y_diff != 0:
                x = F.pad(x, (0, Y_diff, 0, X_diff))
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose2d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        interpolate (bool): if True use nn.Upsample for upsampling, otherwise
            learn ConvTranspose2d if you have enough GPU memory and ain't
            afraid of overfitting
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose2d
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, interpolate, order='crg'):
        super(Decoder, self).__init__()
        if interpolate:
            self.upsample = None
        else:
            # make sure that the output size reverses the MaxPool2d
            # D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0]
            self.upsample = nn.ConvTranspose2d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               output_padding=0)
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=3,
                                      order=order)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            # x = F.upsample(x, scale_factor=self.scale_factor, mode='nearest')
        else:
            x = self.upsample(x)
        # print(x.shape, encoder_features.shape)
        X_diff = encoder_features.shape[2] - x.shape[2]
        Y_diff = encoder_features.shape[3] - x.shape[3]
        if X_diff + Y_diff != 0:
            x = F.pad(x, (0, Y_diff, 0, X_diff))
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


def main():
    # 自己的模型
    model = Res50EncodeUNet(out_channels=2, final_sigmoid=False, interpolate=True, init_channel_number=16)
    print(model)


if __name__ == '__main__':
    main()
