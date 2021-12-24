import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    

class FABlock(nn.Module):
    def __init__(self, in_channels, norm_layer=None, reduction=8):
        super(FABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv1x1(in_channels, 1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.conv2 = conv1x1(in_channels, in_channels)

        self.conv3 = conv1x1(in_channels, 1)
        self.conv4 = conv3x3(1, 1)
        self.bn4 = norm_layer(1)

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        B, C, H, W = x.size()

        # channel attention
        y = self.conv1(x).view(B, 1, -1)
        y = F.softmax(y, dim=-1)
        y = y.permute(0, 2, 1).contiguous()
        y = torch.matmul(x.view(B, C, -1), y).view(B, -1)
        y = self.channel_fc(y)
        y = torch.sigmoid(y).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        x_y = self.conv2(x)
        x_y = x_y * y

        # position attention
        x_y_z = self.conv3(x_y)
        z = self.conv4(x_y_z)
        z = self.bn4(z)
        z = torch.sigmoid(z)
        x_y_z = x_y_z * z

        out = self.gamma*x_y_z + x

        return out