# Reproduction of the article "Deep-learning-enabled geometric constraints and phase unwrapping for single-shot absolute 3D shape measurement"
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        # Assuming x is of shape (batch_size, 4C, H, W)
        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, -1, 2, 2, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, -1, 2 * height, 2 * width)

        return x


class Path1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Path1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.resBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.resBlocks(y)
        y = self.conv2(y)
        return y


class Path2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Path2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.resBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.upSample = UpsampleBlock(64)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxPool(y)
        y = self.resBlocks(y)
        y = self.upSample(y)
        y = self.conv2(y)
        return y


class Path3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Path3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.resBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.upSample = UpsampleBlock(64)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxPool(y)
        y = self.maxPool(y)
        y = self.resBlocks(y)
        y = self.upSample(y)
        y = self.upSample(y)
        y = self.conv2(y)
        return y


class Path4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Path4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.resBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.upSample = UpsampleBlock(64)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxPool(y)
        y = self.maxPool(y)
        y = self.maxPool(y)
        y = self.resBlocks(y)
        y = self.upSample(y)
        y = self.upSample(y)
        y = self.upSample(y)
        y = self.conv2(y)
        return y


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.path1 = Path1(in_channels, 64)
        self.path2 = Path2(in_channels, 64)
        self.path3 = Path3(in_channels, 64)
        self.path4 = Path4(in_channels, 64)
        self.last_conv = nn.Conv2d(64 * 4, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)
        y4 = self.path4(x)
        total = torch.concatenate((y1, y2, y3, y4), dim=1)
        ret = self.last_conv(total)
        return ret


if __name__ == '__main__':
    model = CNN(1, 2)
    x = torch.randn(1, 1, 240, 240)
    ret = model(x)
