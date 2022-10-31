import base.modules as modules
from model.MCResEncodeUNet import *


class TransposeX2(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))

        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            modules.Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            modules.Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class LinknetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            prefinal_channels=16,
            n_blocks=3,
            use_batchnorm=True,
    ):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm)
            for i in range(n_blocks)
        ])

    def forward(self, features):
        # features = features[1:]  # remove first skip
        # features = features[::-1]  # reverse channels to start from head of encoder
        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class MCLinkNet(nn.Module):
    """
    """

    def __init__(self, out_channels, final_sigmoid, interpolate=True, init_channel_number=64):
        super(MCLinkNet, self).__init__()

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = resnet34(ch=init_channel_number)
        self.decoders = LinknetDecoder(
            [init_channel_number, init_channel_number * 2, init_channel_number * 4, init_channel_number * 8])

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
        _, encoders_features = self.encoders(x)

        x = self.decoders(encoders_features)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x


def main():
    net = LinkNet(out_channels=2, final_sigmoid=False, interpolate=True, init_channel_number=16)
    print(net)


if __name__ == '__main__':
    main()
