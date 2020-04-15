import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample1 = nn.Upsample(size=62)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.upsample2 = nn.Upsample(size=125)

        self.dconv_up2 = double_conv(128 + 256, 128)
        self.upsample3 = nn.Upsample(size=250)

        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 250
        x = self.maxpool(conv1)    # 125

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)    # 62

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)    # 31

        x = self.dconv_down4(x)

        x = self.upsample1(x)       # 62
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample2(x)       # 124
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class CNN(UNet):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7936, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 4)
        self.softmax = nn.Softmax()

    def forward(self, x):
        conv1 = self.dconv_down1(x)  # 250
        x = self.maxpool(conv1)    # 125

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)    # 62

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)    # 31

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
