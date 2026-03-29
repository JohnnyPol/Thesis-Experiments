import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd

class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
    self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(out_channels))
    self.downsample = downsample
    self.relu = nn.ReLU()
    self.out_channels = out_channels

  def forward(self, x):
      residual = x
      out = self.conv1(x)
      out = self.conv2(out)
      if self.downsample:
        residual = self.downsample(x)
      out += residual
      out = self.relu(out)
      return out
  
class ResidualBlock50(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1):
        super(ExitBlock, self).__init__()
        layers = []
        channels = in_channels

        for _ in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)
    
class ExitBlock50(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1, reduction=0.25):
        super(ExitBlock50, self).__init__()

        reduced_channels = max(16, int(in_channels * reduction))
        layers = []

        if num_convs > 0:
            layers.append(nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1))
            layers.append(nn.BatchNorm2d(reduced_channels))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs):
                layers.append(nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(reduced_channels))
                layers.append(nn.ReLU(inplace=True))

            self.features = nn.Sequential(*layers)
            classifier_in = reduced_channels
        else:
            self.features = nn.Identity()
            classifier_in = in_channels

        self.classifier = nn.Linear(classifier_in, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)