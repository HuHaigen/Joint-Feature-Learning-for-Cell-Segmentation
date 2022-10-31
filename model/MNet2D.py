import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from model.basic import Encoder, Decoder, create_feature_maps, ExtResNetBlock


class MNet2D(nn.Module):
    """
    
    """

    def __init__(self, in_channels, out_channels,
                 final_sigmoid,
                 conv_kernel_size=3,
                 trans_kernel_size=3,
                 scale_factor=2,
                 max_pool_kernel_size=2,
                 max_pool_strides=2,
                 interpolate=True,
                 conv_layer_order='cgr',
                 init_channel_number=64,
                 model_deep=3,
                 M_channel=2,
                 M_R=0):
        super(MNet2D, self).__init__()
        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        if model_deep == 3:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                        max_pool_strides=1, conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(4 * init_channel_number, 8 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R)
            ])

            self.decoders = nn.ModuleList([
                Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
            ])
        elif model_deep == 2:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                        max_pool_strides=1, conv_layer_order=conv_layer_order, M_channel=M_channel),
                Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel),
                Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel)
            ])

            self.decoders = nn.ModuleList([
                Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel),
                Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel)
            ])
        elif model_deep == 1:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                        max_pool_strides=1, conv_layer_order=conv_layer_order, M_channel=M_channel),
                Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel)
            ])

            self.decoders = nn.ModuleList([
                Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel)
            ])
        if model_deep == 4:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                        max_pool_strides=1, conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(4 * init_channel_number, 8 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(8 * init_channel_number, 16 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R)
            ])

            self.decoders = nn.ModuleList([
                Decoder(8 * init_channel_number + 16 * init_channel_number, 8 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
            ])
        if model_deep == 5:
            self.encoders = nn.ModuleList([
                Encoder(in_channels, init_channel_number, conv_kernel_size=conv_kernel_size, is_max_pool=False,
                        max_pool_strides=1, conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(init_channel_number, 2 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(2 * init_channel_number, 4 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(4 * init_channel_number, 8 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(8 * init_channel_number, 16 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Encoder(16 * init_channel_number, 32 * init_channel_number, conv_kernel_size=conv_kernel_size,
                        is_max_pool=True,
                        max_pool_kernel_size=max_pool_kernel_size, max_pool_strides=max_pool_strides,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R)
            ])

            self.decoders = nn.ModuleList([
                Decoder(16 * init_channel_number + 32 * init_channel_number, 16 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(8 * init_channel_number + 16 * init_channel_number, 8 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
                Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                        trans_kernel_size=trans_kernel_size, scale_factor=scale_factor,
                        conv_layer_order=conv_layer_order, M_channel=M_channel, M_R=M_R),
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

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing accuracy
        if not self.training:
            x = self.final_activation(x)
        return x


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm2d+ReLU+Conv2d)
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv2d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv2d+BatchNorm2d+ReLU use order='cbr'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'cgr' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cgr'):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size, order)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size, "cg")

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order):
        """Add the conv layer with non-linearity and optional batchnorm

        Args:
            pos (int): the order (position) of the layer. MUST be 1 or 2
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            order (string): order of things, e.g.
                'cr' -> conv + ReLU
                'cgr' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """
        assert pos in [1, 2], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r' or 's' in order, "'r' (ReLU layer) MUST be present"
        assert order[0] is not 'r', 'ReLU cannot be the first operation in the layer'

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('relu{}'.format(pos), nn.ReLU(inplace=True))
            elif char == 's':
                self.add_module('sigmoid{}'.format(pos), nn.Sigmoid())
            elif char == 'c':
                self.add_module('conv{}'.format(pos), nn.Conv2d(in_channels,
                                                                out_channels,
                                                                kernel_size,
                                                                padding=((kernel_size - 1) // 2)))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm3d MUST go after the Conv2d'
                # self.add_module('norm{}'.format(pos), groupnorm.GroupNorm3d(out_channels))
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

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, is_max_pool=True,
                 max_pool_kernel_size=3, max_pool_strides=1, conv_layer_order='cgr', M_channel=2, M_R=0):
        super(Encoder, self).__init__()
        self.max_pool_size = max_pool_kernel_size
        self.M_channel = M_channel
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_strides,
                                     padding=(max_pool_kernel_size - 1) // 2) if is_max_pool else None
        if M_R == 0:
            self.double_conv1 = DoubleConv(in_channels, out_channels, kernel_size=1, order=conv_layer_order)
            self.double_conv3 = DoubleConv(in_channels, out_channels, kernel_size=3, order=conv_layer_order)
            if self.M_channel == 3:
                self.double_conv5 = DoubleConv(in_channels, out_channels, kernel_size=5, order=conv_layer_order)
        else:
            self.double_conv1 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=3, order=conv_layer_order)
            self.double_conv3 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=3, order=conv_layer_order)
            if self.M_channel == 3:
                self.double_conv5 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=5, order=conv_layer_order)

        # self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x1 = self.double_conv1(x)
        x3 = self.double_conv3(x)

        if self.M_channel == 3:
            x5 = self.double_conv5(x)
            x = x1 + x3 + x5
        else:
            x = x1 + x3
        # x = self.conv1(x)
        x = self.relu(x)

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

    def __init__(self, in_channels, out_channels, interpolate, trans_kernel_size=3,
                 scale_factor=1, conv_layer_order='cgr', M_channel=2, M_R=0):
        super(Decoder, self).__init__()
        self.scale_factor = scale_factor
        self.M_channel = M_channel
        if interpolate:
            self.upsample = None
        else:
            # make sure that the output size reverses the MaxPool2d
            # D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0]
            self.upsample = nn.ConvTranspose2d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=trans_kernel_size,
                                               stride=scale_factor,
                                               padding=((trans_kernel_size - 1) // 2),
                                               output_padding=scale_factor - 1)
        if M_R==0:
            self.double_conv1 = DoubleConv(in_channels, out_channels, kernel_size=1, order=conv_layer_order)
            self.double_conv3 = DoubleConv(in_channels, out_channels, kernel_size=3, order=conv_layer_order)
            if self.M_channel == 3:
                self.double_conv5 = DoubleConv(in_channels, out_channels, kernel_size=5, order=conv_layer_order)
        else:
            self.double_conv1 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=3, order=conv_layer_order)
            self.double_conv3 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=3, order=conv_layer_order)
            if self.M_channel == 3:
                self.double_conv5 = ExtResNetBlock(in_channels, out_channels, num_groups=1, kernel_size=5, order=conv_layer_order)

        # self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

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
        x1 = self.double_conv1(x)
        x3 = self.double_conv3(x)
        if self.M_channel == 3:
            x5 = self.double_conv5(x)
            x = x1 + x3 + x5
        else:
            x = x1 + x3
        # x = self.conv1(x)
        x = self.relu(x)
        return x


def main():
    model = MNet2D(in_channels=1, out_channels=2, init_channel_number=1,
                   conv_layer_order='cgr',
                   interpolate=False,
                   final_sigmoid=False)
    data = torch.zeros((1, 1, 48, 48))
    out = model(data)
    print(model)


if __name__ == '__main__':
    main()
