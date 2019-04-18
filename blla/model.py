import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

from blla import darknet


class FeatureExtractionNet(nn.Module):
    """
    Feature extraction from 53-layer darknet.
    """
    def __init__(self, pretrained):
        super().__init__()
        self.dk = pretrained.feat[:-1]

    def forward(self, inputs):
        siz = inputs.size()
        # downsampled by 2
        ds_2 = self.dk[:7](inputs)
        ds_3 = self.dk[7:12](ds_2)
        ds_4 = self.dk[12:23](ds_3)
        ds_5 = self.dk[23:34](ds_4)
        ds_6 = self.dk[34:41](ds_5)

        return ds_2, ds_3, ds_4, ds_5, ds_6


class UnetDecoder(nn.Module):
    """
    U-Net decoder block with a convolution before upsampling.
    """
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.deconv = nn.ConvTranspose2d(inter_channels, out_channels, 3, padding=1, stride=2)
        self.norm_act = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x, output_size):
        x = self.layer(x)
        return self.norm_act(self.deconv(x, output_size=output_size))

class VerticeNet(nn.Module):
    """
    Initial vertices calculation from features.
    """
    def __init__(self):
        super().__init__()
        # initial vertices layer
        self.upsample_6 = UnetDecoder(1024, 512, 256)
        self.upsample_5 = UnetDecoder(768, 384, 192)
        self.upsample_4 = UnetDecoder(448, 224, 112)
        self.upsample_3 = UnetDecoder(240, 120, 60)
        self.squash = nn.Sequential(nn.Conv2d(124, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    nn.Conv2d(128, 1, 1, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.Sigmoid())

    def forward(self, ds_2, ds_3, ds_4, ds_5, ds_6):
        map_5 = self.upsample_6(ds_6, output_size=ds_5.size())
        map_4 = self.upsample_5(torch.cat([map_5, ds_5], 1), output_size=ds_4.size())
        map_3 = self.upsample_4(torch.cat([map_4, ds_4], 1), output_size=ds_3.size())
        map_2 = self.upsample_3(torch.cat([map_3, ds_3], 1), output_size=ds_2.size())
        return self.squash(torch.cat([map_2, ds_2], 1))

class FeatureNet(nn.Module):
    """
    Upsamples and reduces dimensionality of feature maps from darknet
    """
    def __init__(self):
        super().__init__()
        self.upsample_6 = UnetDecoder(1024, 512, 128)
        self.upsample_5 = UnetDecoder(640, 320, 128)
        self.upsample_4 = UnetDecoder(384, 192, 128)
        self.upsample_3 = UnetDecoder(256, 128, 128)
        self.squash = nn.Sequential(nn.Conv2d(192, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.1))

    def forward(self, ds_2, ds_3, ds_4, ds_5, ds_6):
        map_5 = self.upsample_6(ds_6, output_size=ds_5.size())
        map_4 = self.upsample_5(torch.cat([map_5, ds_5], 1), output_size=ds_4.size())
        map_3 = self.upsample_4(torch.cat([map_4, ds_4], 1), output_size=ds_3.size())
        map_2 = self.upsample_3(torch.cat([map_3, ds_3], 1), output_size=ds_2.size())
        return self.squash(torch.cat([map_2, ds_2], 1))

class PolyLineNet(nn.Module):
    """
    2D separable LSTM predicting next vertex in polyline based on last 3
    vertices and a feature map.

    """
    def __init__(self, in_channels, out_channels, bidi=True):
        super().__init__()
        self.bidi = bidi
        self.hidden_size = out_channels
        self.output_size = out_channels if not self.bidi else 2*out_channels
        self.hrnn = nn.LSTM(in_channels, self.hidden_size, batch_first=True, bidirectional=bidi)
        self.vrnn = nn.LSTM(self.output_size, out_channels, batch_first=True, bidirectional=bidi)

    def forward(self, inputs):
        # NCHW -> HNWC
        inputs = inputs.permute(2, 0, 3, 1)
        siz = inputs.size()
        # HNWC -> (H*N)WC
        inputs = inputs.contiguous().view(-1, siz[2], siz[3])
        # (H*N)WO
        o, _ = self.hrnn(inputs)
        # resize to HNWO
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        # vertical pass
        # HNWO -> WNHO
        o = o.transpose(0, 2)
        # (W*N)HO
        o = o.view(-1, siz[0], self.output_size)
        # (W*N)HO'
        o, _ = self.vrnn(o)
        # (W*N)HO' -> WNHO'
        o = o.view(siz[2], siz[1], siz[0], self.output_size)
        # WNHO' -> NO'HW
        return torch.sum(o.permute(1, 3, 2, 0), dim=1).unsqueeze(1)


