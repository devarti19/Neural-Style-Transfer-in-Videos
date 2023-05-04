import torch.nn as nn
from math import floor

class ConvLayer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        padding = int(floor(kernel_size / 2))
        self.pad = nn.ReflectionPad2d(padding)
        self.conv_layer = nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        return x


class ConvInstRelu(ConvLayer):
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        super(ConvInstRelu, self).__init__(input_channel, output_channel, kernel_size, stride)
        self.inst_norm = nn.InstanceNorm2d(output_channel, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super(ConvInstRelu, self).forward(x)
        x = self.inst_norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding=padding)
        self.inst_norm1 = nn.InstanceNorm2d(output_channel, affine=True)
        self.conv_layer2 = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding=padding)
        self.inst_norm2 = nn.InstanceNorm2d(output_channel, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_layer1(x)
        x = self.inst_norm1(x)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.inst_norm2(x)
        x = residual + x
        return x

class ConvTanh(ConvLayer):
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        super(ConvTanh, self).__init__(input_channel, output_channel, kernel_size, stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = super(ConvTanh, self).forward(x)
        x = self.tanh(x)
        return x


class ImageTransformer(nn.Module):
    def __init__(self):
        super(ImageTransformer, self).__init__()
        self.enc1 = ConvInstRelu(3, 32, 9, 1)
        self.enc2 = ConvInstRelu(32, 64, 3, 2)
        self.enc3 = ConvInstRelu(64, 128, 3, 2)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deco1 = ConvInstRelu(128, 64, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deco2 = ConvInstRelu(64, 32, 3, 1)
        self.deco3 = ConvTanh(32, 3, 9, 1)

    def forward(self, x):
        #encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        #residual
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        #decoder
        x = self.upsample1(x)
        x = self.deco1(x)
        x = self.upsample2(x)
        x = self.deco2(x)
        x = self.deco3(x)
        return x