import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

class ResBlock(nn.Module):
    def __init__(self, filter_multiplier=1):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(filter_multiplier*64, filter_multiplier*32, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(filter_multiplier*32),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(filter_multiplier*32, filter_multiplier*64, kernel_size=(3, 3), padding=1, bias=False),
                                   nn.BatchNorm2d(filter_multiplier*64),
                                   nn.LeakyReLU(negative_slope=0.1))


    def forward(self, x):
        o = self.layer(x)
        return o + x

class Darknet53(nn.Module):
    """
    Darknet 53 implementation.
    """
    def __init__(self):
        super().__init__()

        self.feat = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  ResBlock(1),
                                  nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  ResBlock(2),
                                  ResBlock(2),
                                  nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  ResBlock(4),
                                  nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  ResBlock(8),
                                  nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
                                  nn.BatchNorm2d(1024),
                                  nn.LeakyReLU(negative_slope=0.1),
                                  ResBlock(16),
                                  ResBlock(16),
                                  ResBlock(16),
                                  ResBlock(16),
                                  nn.AdaptiveAvgPool2d(1))
        self.lin = nn.Conv2d(1024, 1000, kernel_size=1)

    def forward(self, x):
        o = self.feat(x)
        o = self.lin(o)
        return o

