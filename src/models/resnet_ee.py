import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.blocks import ExitBlock, ExitBlock50

class ResNetEE18(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9):
        super(ResNetEE18, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.exit0 = ExitBlock(64, num_classes, num_convs=3)
        self.exit1 = ExitBlock(128, num_classes, num_convs=2)
        self.exit2 = ExitBlock(256, num_classes, num_convs=1)


        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)

            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.layer0(x)
          out0 = self.exit0(x)
          if self._confident_enough(out0):
            return out0, 0

          x = self.layer1(x)
          out1 = self.exit1(x)
          if self._confident_enough(out1):
            return out1, 1

          x = self.layer2(x)
          out2 = self.exit2(x)
          if self._confident_enough(out2):
            return out2, 2

          x = self.layer3(x)

          xf = self.avgpool(x)
          xf = torch.flatten(xf, 1)
          out_final = self.fc(xf)
          return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.item() <= self.confidence_threshold


class ResNetEE50(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9):
        super(ResNetEE50, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.exit0 = ExitBlock50(64 * block.expansion, num_classes, num_convs=3)
        self.exit1 = ExitBlock50(128 * block.expansion, num_classes, num_convs=2)
        self.exit2 = ExitBlock50(256 * block.expansion, num_classes, num_convs=1)

        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)

            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.layer0(x)
          out0 = self.exit0(x)
          if self._confident_enough(out0):
            return out0, 0

          x = self.layer1(x)
          out1 = self.exit1(x)
          if self._confident_enough(out1):
            return out1, 1

          x = self.layer2(x)
          out2 = self.exit2(x)
          if self._confident_enough(out2):
            return out2, 2

          x = self.layer3(x)

          xf = self.avgpool(x)
          xf = torch.flatten(xf, 1)
          out_final = self.fc(xf)
          return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.item() <= self.confidence_threshold