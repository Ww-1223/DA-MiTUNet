import torch
import torch.nn as nn
import math


class ECABlock(nn.Module):
    def __init__(self, img_channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(img_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, channels, ratio=2):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)
        return x * v

class Channel_Attention_Module(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Channel_Attention_Module, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v

class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int = 7):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        # print(avg_x.shape)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        # print(max_x.shape)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        # print(v.shape)
        v = self.sigmoid(v)
        return x * v


class CBAMBlock(nn.Module):
    def __init__(self, spatial_attention_kernel_size: int = 3, channels: int = None, gamma: int = 2, b: int = 1):
        super(CBAMBlock, self).__init__()
        # self.channel_attention_block = Channel_Attention_Module(channels=channels, gamma=gamma, b=b)
        self.channel_attention_block_fc = Channel_Attention_Module_FC(channels=channels)
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1,1,0,bias=False),
            # nn.GroupNorm(12, channels, eps=1e-6),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        # out = self.channel_attention_block(x)
        out = self.channel_attention_block_fc(x)
        out = self.spatial_attention_block(out)
        out = self.conv(torch.cat((x,out), dim=1))
        return out


if __name__ == '__main__':
    input = torch.randn((1,3,512,512))
    # out = Spatial_Attention_Module(k=7)(input)
    out = CBAMBlock(channels=3)(input)
    print(out.shape)
    a = torch.randn((1,1,3))
    b = torch.randn((1,1,3))
    mul = torch.mul(a,b)

    print(mul.shape)
    print(mul)