import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

from blla import darknet


class FeatureNet(nn.Module):
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

        # upsample into vector of size (N*1984*H/2*W/2)
        feat = torch.cat([ds_2,
                          F.interpolate(ds_3, size=ds_2.shape[2:]),
                          F.interpolate(ds_4, size=ds_2.shape[2:]),
                          F.interpolate(ds_5, size=ds_2.shape[2:]),
                          F.interpolate(ds_6, size=ds_2.shape[2:])], dim=1)
        return feat


class VerticeNet(nn.Module):
    """
    Initial vertices calculation from features.
    """
    def __init__(self, refine_encoder=False):
        super().__init__()
        # initial vertices layer
        self.vertices_conv = nn.Sequential(nn.Conv2d(1984, 128, 1, bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.LeakyReLU(negative_slope=0.1),
                                           nn.Conv2d(128, 256, 3, padding=2, dilation=2, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.LeakyReLU(negative_slope=0.1),
                                           nn.Conv2d(256, 128, 3, padding=3, dilation=4, bias=False),
                                           nn.BatchNorm2d(128),
                                           nn.LeakyReLU(negative_slope=0.1),
                                           nn.Conv2d(128, 1, 1, padding=1),
                                           nn.Sigmoid())

    def forward(self, inputs):
        # endpoint vertices predicition network
        o = self.vertices_conv(feat)
        return o


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
