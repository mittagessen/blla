import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for p in m.parameters():
            # weights
            if p.data.dim() == 2:
                nn.init.orthogonal_(p.data)
            # initialize biases to 1 (jozefowicz 2015)
            else:
                nn.init.constant_(p.data[len(p)//4:len(p)//2], 1.0)
    elif isinstance(m, nn.GRU):
        for p in m.parameters():
            nn.init.orthogonal_(p.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


class UnetDecoder(nn.Module):
    """
    U-Net decoder block with a convolution before upsampling.
    """
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(inter_channels, out_channels, 3, padding=1, stride=2)
        self.norm_conv = nn.GroupNorm(32, inter_channels)
        self.norm_deconv = nn.GroupNorm(32, out_channels)

    def forward(self, x, output_size):
        x = F.relu(self.norm_conv(self.conv(x)))
        return F.relu(self.norm_deconv(self.deconv(x, output_size=output_size)))

class ResUNet(nn.Module):
    """
    ResNet-34 encoder + U-Net decoder
    """
    def __init__(self, refine_encoder=False):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        if not refine_encoder:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        # operating on map_4
        self.upsample_4 = UnetDecoder(256, 256, 128)
        # operating on cat(map_3, upsample_5(map_4))
        self.upsample_3 = UnetDecoder(256, 128, 64)
        self.upsample_2 = UnetDecoder(128, 64, 64)
        self.upsample_1 = UnetDecoder(128, 64, 64)
        self.squash = nn.Conv2d(64, 1, kernel_size=1)

        self.nonlin = nn.Sigmoid()
        #self.init_weights()

    def forward(self, inputs):
        siz = inputs.size()
        x = self.resnet.conv1(inputs)
        x = self.resnet.bn1(x)
        map_1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        map_2 = self.resnet.layer1(x)
        map_3 = self.resnet.layer2(map_2)
        map_4 = self.resnet.layer3(map_3)

        # upsample concatenated maps
        map_4 = self.dropout(self.upsample_4(map_4, output_size=map_3.size()))
        map_3 = self.dropout(self.upsample_3(torch.cat([map_3, map_4], 1), output_size=map_2.size()))
        map_2 = self.dropout(self.upsample_2(torch.cat([map_2, map_3], 1), output_size=map_1.size()))
        map_1 = self.dropout(self.upsample_1(torch.cat([map_1, map_2], 1), output_size=map_1.size()[:2] + siz[2:]))
        return self.squash(map_1)

    def init_weights(self):
        self.upsample_4.apply(weight_init)
        self.upsample_3.apply(weight_init)
        self.upsample_2.apply(weight_init)
        self.upsample_1.apply(weight_init)
        self.squash.apply(weight_init)

