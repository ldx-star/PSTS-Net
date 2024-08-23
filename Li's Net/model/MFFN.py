 # Reproduction of the article "Absolute phase retrieval for a single-shot fringe projection profilometry based on deep learning"
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        x1 = y
        y = self.conv3(y)
        y = self.relu(y)
        y = self.conv4(y)
        y = self.relu(y)
        ret = x + x1 + y
        return ret


class Sequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Sequence, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.resBlock = ResidualBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.resBlock(y)
        y = self.conv2(y)
        y = self.relu(y)
        return y


class MFFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFFN, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.Sequence1 = Sequence(in_channels, 64)
        self.Sequence2 = Sequence(64, 64)
        self.Sequence3 = Sequence(64, 64)
        self.Sequence4 = Sequence(64, 64)
        self.Sequence5 = Sequence(64, out_channels)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.resBlock = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        e1 = self.Sequence1(x)
        in2 = self.conv1(self.maxpool(x))
        y2 = self.Sequence2(in2)
        in3 = self.maxpool(in2)
        y3 = self.Sequence3(in3)
        in4 = self.maxpool(in3)
        y4 = self.Sequence4(in4)

        e2 = self.upsample1(y2)
        e3 = self.upsample2(y3)
        e4 = self.upsample3(y4)

        e5 = e1 + e2 + e3 + e4

        y = self.conv2(e5)
        y = self.resBlock(y)
        y = self.conv3(y)
        return y


if __name__ == '__main__':
    x = torch.rand(1, 1, 240, 240)
    model = MFFN(in_channels=1, out_channels=2)
    y1, y2 = model(x)