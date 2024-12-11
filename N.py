import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from thop import profile

from lib.pvtv2 import pvt_v2_b2
import numpy as np
from time import time


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):#[1, 32, 22, 22]
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x): ## x1[1, 320, 22, 22]
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result  # x1[1, 512, 11, 11]







class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'C:/Users/Lenovo/Desktop/MSCAF-COD-master/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.cbam3 = CBAM(512)
        self.cbam2 = CBAM(320)
        self.cbam1 = CBAM(128)
        self.cbam0 = CBAM(64)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)

        x1 = pvt[0]# x1[1, 64, 88, 88]
        x2 = pvt[1]# x2[1, 128, 44, 44]
        x3 = pvt[2]# x3[1, 320, 22, 22]
        x4 = pvt[3]# x4[1, 512, 11, 11]

        c3 = self.cbam3(x4)   #c3[1, 512, 11, 11]
        c2 = self.cbam2(x3) #c2[1, 320, 22, 22]
        c1 = self.cbam1(x2) #c1[1, 128, 44, 44]
        c0 = self.cbam0(x1) #c0 [1, 64, 88, 88]

        c3 = self.Translayer4_1(c3)
        c2 = self.Translayer3_1(c2)
        c1 = self.Translayer2_1(c1)
        c0 = self.Translayer2_0(c0)

        c3 = self.linearr4(c3)
        c2 = self.linearr3(c2)
        c1 = self.linearr2(c1)
        c0 = self.linearr1(c0)
        
        out_3 = F.interpolate(c3, size=image_shape, mode='bilinear')
        out_2 = F.interpolate(c2, size=image_shape, mode='bilinear')
        out_1 = F.interpolate(c1, size=image_shape, mode='bilinear')
        out_0 = F.interpolate(c0, size=image_shape, mode='bilinear')

        return out_3, out_2, out_1, out_0

if __name__ == '__main__':

    net = Network().cuda()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352).cuda()
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        # print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    flops, params = profile(net, (dump_x,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024 * 1024 * 1024), params / (1024 * 1024)))
    # print(np.mean(frame_rate))
    # print(y)