from .Linknet import LinkNet
from .MCLinknet import MCLinkNet
from .MCVGGEncodeUnet import MCVGGEncodeUnet
from .MNet2D import MNet2D
from .ResEncodeUNet import ResEncodeUNet
from .Unet import UNet
from .UResNet import ResidualUNet2D
from .VGGEncodeUnet import VGGEncodeUnet


models = ["UNET", "MCUNET", "RESUNET", "Res50EncodeUNet", "RES101EncodeUNet",
          "VGGEncodeUnet", "MCVGGEncodeUnet", "LinkNet", "MCLinkNet"]


def build_model(cfg):
    model_name = cfg['model']
    assert model_name in models, "Unsupported model..."

    if cfg['model'] == "UNET":
        model = UNet(in_channels=cfg['in_channels'],
                     out_channels=cfg['out_channels'],
                     init_channel_number=cfg['init_channel_number'],
                     interpolate=cfg['interpolate'],
                     final_sigmoid=cfg['final_sigmoid'],
                     model_deep=cfg['model_deep'])
    elif cfg['model'] == "MCUNET":
        model = MNet2D(in_channels=cfg['in_channels'],
                       out_channels=cfg['out_channels'],
                       init_channel_number=cfg['init_channel_number'],
                       interpolate=cfg['interpolate'],
                       final_sigmoid=cfg['final_sigmoid'],
                       model_deep=cfg['model_deep'],
                       M_channel=cfg['M_channel'],
                       M_R=cfg['M_R'])
    elif cfg['model'] == "RESUNET":
        model = ResidualUNet2D(in_channels=cfg['in_channels'],
                               out_channels=cfg['out_channels'],
                               final_sigmoid=cfg['final_sigmoid'],
                               f_maps=cfg['init_channel_number'],
                               conv_layer_order='cgr',
                               num_groups=1,
                               model_deep=cfg['model_deep'])
    elif cfg['model'] == "Res50EncodeUNet":
        model = ResEncodeUNet(out_channels=cfg['out_channels'],
                              final_sigmoid=cfg['final_sigmoid'],
                              interpolate=cfg['interpolate'],
                              init_channel_number=cfg['init_channel_number'])
    elif cfg['model'] == "RES101EncodeUNet":
        model = ResEncodeUNet(out_channels=cfg['out_channels'],
                              final_sigmoid=cfg['final_sigmoid'],
                              interpolate=cfg['interpolate'],
                              init_channel_number=cfg['init_channel_number'],
                              resname="101")
    elif cfg['model'] == "VGGEncodeUnet":
        model = VGGEncodeUnet(out_channels=cfg['out_channels'],
                              final_sigmoid=cfg['final_sigmoid'],
                              interpolate=cfg['interpolate'],
                              init_channel_number=cfg['init_channel_number'])
    elif cfg['model'] == "MCVGGEncodeUnet":
        model = MCVGGEncodeUnet(out_channels=cfg['out_channels'],
                                final_sigmoid=cfg['final_sigmoid'],
                                interpolate=cfg['interpolate'],
                                init_channel_number=cfg['init_channel_number'])
    elif cfg['model'] == "LinkNet":
        model = LinkNet(out_channels=cfg['out_channels'],
                        final_sigmoid=cfg['final_sigmoid'],
                        interpolate=cfg['interpolate'],
                        init_channel_number=cfg['init_channel_number'])
    elif cfg['model'] == "MCLinkNet":
        model = MCLinkNet(out_channels=cfg['out_channels'],
                          final_sigmoid=cfg['final_sigmoid'],
                          interpolate=cfg['interpolate'],
                          init_channel_number=cfg['init_channel_number'])

    return model
