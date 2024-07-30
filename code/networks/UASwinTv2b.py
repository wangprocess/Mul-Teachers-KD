from __future__ import division, print_function

import cv2

from .SKAttention import SKAttention
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_V2_B_Weights


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownBlock_with_SKAttention(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock_with_SKAttention, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)
        self.SKAttention = SKAttention(channel=in_channels, reduction=8)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.SKAttention(x1)
        x_merge = x1 + x2
        x_out = self.conv(x_merge)
        return x_out


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock_with_SKAttention(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock_with_SKAttention(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock_with_SKAttention(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock_with_SKAttention(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.in_chns,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UASwinV2B(nn.Module):
    def __init__(self, in_chns, num_classes, image_size):
        super(UASwinV2B, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.swin_v2 = models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        for param in self.swin_v2.parameters():
            param.requires_grad = True
        self.swin_v2.head = nn.Sequential(nn.Dropout(p=0.2),
                                          nn.Linear(1024, num_classes))

    def forward(self, x):
        feature = self.encoder(x)
        feature2 = self.decoder(feature)
        output = self.swin_v2(feature2)
        return output


if __name__ == '__main__':
    # Hyperparameters
    img_size = 224
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 10
    num_classes = 7
    channels = 3

    # Initialize model
    model = UASwinV2B(channels, num_classes, img_size).cuda()
    # print(model)

    # 计算参数量
    params_num = sum(p.numel() for p in model.parameters())
    print("\nModle's Params: %.3fM" % (params_num / 1e6))
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()

    output = model(x)
    print(model)
    print(output.size())
