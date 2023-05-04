import torch.nn as nn
import torchvision


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True, progress=True).features.eval()
        self.s1 = nn.Sequential()
        self.s2 = nn.Sequential()
        self.s3 = nn.Sequential()
        self.s4 = nn.Sequential()

        for i in range(4):
            self.s1.add_module(str(i), vgg[i])
        for i in range(4, 9):
            self.s2.add_module(str(i), vgg[i])
        for i in range(9, 16):
            self.s3.add_module(str(i), vgg[i])
        for i in range(16, 23):
            self.s4.add_module(str(i), vgg[i])

    def forward(self, x):

        x = self.s1(x)
        relu1_2 = x
        x = self.s2(x)
        relu2_2 = x
        x = self.s3(x)
        relu3_3 = x
        x = self.s4(x)
        relu4_3 = x

        return relu1_2, relu2_2, relu3_3, relu4_3


class Normal(nn.Module):
    def __init__(self, mean, std):
        super(Normal, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std