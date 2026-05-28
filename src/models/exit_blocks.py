import torch.nn as nn
import torch.nn.functional as F

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
