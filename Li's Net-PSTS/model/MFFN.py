# PSTS based on Li'Net
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


class MFFN1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFFN1, self).__init__()
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
        e2 = self.Sequence2(in2)
        in3 = self.maxpool(in2)
        e3 = self.Sequence3(in3)
        in4 = self.maxpool(in3)
        e4 = self.Sequence4(in4)

        y2 = self.upsample1(e2)
        y3 = self.upsample2(e3)
        y4 = self.upsample3(e4)

        e5 = e1 + y2 + y3 + y4

        e5 = self.conv2(e5)
        y = self.resBlock(e5)
        y = self.conv3(y)
        return y, e2, e3, e4, e5


class MFFN2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFFN2, self).__init__()
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

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.resBlock = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.fuature_fusion1 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        self.fuature_fusion2 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        self.fuature_fusion3 = nn.Conv2d(64 * 2, 64, kernel_size=1)

        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

    def forward(self, x, x2, x3, x4):
        e1 = self.Sequence1(x)
        in2 = self.maxpool(x)
        y2 = self.Sequence2(self.fuature_fusion1(torch.concatenate((in2, x2), 1)))
        in3 = self.maxpool(in2)
        y3 = self.Sequence3(self.fuature_fusion2(torch.concatenate((in3, x3), 1)))
        in4 = self.maxpool(in3)
        y4 = self.Sequence4(self.fuature_fusion3(torch.concatenate((in4, x4), 1)))

        e2 = self.upsample1(y2)
        e3 = self.upsample2(y3)
        e4 = self.upsample3(y4)

        e5 = e1 + e2 + e3 + e4

        y = self.conv2(e5)
        y = self.resBlock(y)
        y = self.conv3(y)
        return y


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3 = conv(n_feat, 1, kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv3(self.relu(x1)))
        x_img = x_img * x2
        return x_img


class PSTT_MFFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSTT_MFFN, self).__init__()
        self.mffn1 = MFFN1(in_channels, 11)
        self.mffn2 = MFFN2(64, out_channels)
        self.sam = SAM(64, 3, False)
        self.conv1 = conv(1, 64, 1)

    def forward(self, x):
        y1, e2, e3, e4, e5 = self.mffn1(x)
        sam = self.sam(e5, self.conv1(x))
        y2 = self.mffn2(sam, e2, e3, e4)
        return y1, y2


if __name__ == '__main__':
    x = torch.rand(1, 1, 240, 240)
    model = PSTT_MFFN(in_channels=1, out_channels=2)
    y1, y2 = model(x)
