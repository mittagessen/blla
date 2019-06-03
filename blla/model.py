import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

class ReNet(nn.Module):
    """
    Recurrent block from ReNet.

    Performs a horizontal pass over input features, followed by a vertical pass
    over the output of the first pass.
    """
    def __init__(self, in_channels, out_channels, bidi=True):
        super().__init__()
        self.bidi = bidi
        self.hidden_size = out_channels
        self.output_size = out_channels if not self.bidi else 2*out_channels
        self.hrnn = nn.LSTM(in_channels, self.hidden_size, batch_first=True, bidirectional=bidi)
        self.vrnn = nn.LSTM(self.output_size, out_channels, batch_first=True, bidirectional=bidi)

    def forward(self, inputs):
        # horizontal pass
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
        return o.permute(1, 3, 2, 0)


class RecLabelNet(nn.Module):
    """
    separable recurrent net for baseline labeling.
    """
    def __init__(self, sigmoid=True):
        super().__init__()
        self.label = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False),
                                   nn.GroupNorm(32, 128),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(128, 64, 3, padding=1, stride=2, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   ReNet(64, 32),
                                   nn.Conv2d(64, 32, 1, bias=False),
                                   nn.GroupNorm(32, 32),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   ReNet(32, 32),
                                   nn.Conv2d(64, 1, 1, bias=False))
        if sigmoid:
            self.label.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        siz = x.size()
        o = self.label(x)
        return F.interpolate(o, size=(siz[2], siz[3]))

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
    def __init__(self, refine_encoder=False, sigmoid=True, rnn=False):
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
        self.rnn_stack = nn.Sequential()
        if rnn:
            self.rnn_stack.add_module('rnn', ReNet(64, 16))
            self.rnn_stack.add_module('dropout', nn.Dropout(0.5))
            self.rnn_stack.add_module('squash', nn.Conv2d(32, 1, kernel_size=1))
        else:
            self.rnn_stack.add_module('squash', nn.Conv2d(64, 1, kernel_size=1))
        if sigmoid:
            self.rnn_stack.add_module('sigmoid', nn.Sigmoid())

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
        return self.rnn_stack(map_1)

    def init_weights(self):
        pass
