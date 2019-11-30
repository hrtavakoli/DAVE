#
# DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction
# https://arxiv.org/abs/1905.10693
# https://hrtavakoli.github.io/DAVE/
#
# Copyright by Hamed Rezazadegan Tavakoli
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.resnet3D import resnet18


class ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(ScaleUp, self).__init__()

        self.combine = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)

        self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.combine.weight)
        nn.init.constant_(self.combine.bias, 0.0)

    def forward(self, inputs):
        output = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.combine(output)
        output = F.relu(output, inplace=True)
        return output


class DAVE(nn.Module):

    def __init__(self):
        super(DAVE, self).__init__()

        self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=16, num_classes=12, last_fc=False, last_pool=True)
        self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False, last_pool=False)
        self.upscale1 = ScaleUp(512, 512)
        self.upscale2 = ScaleUp(512, 128)
        self.combinedEmbedding = nn.Conv2d(1024, 512, kernel_size=1)
        self.saliency = nn.Conv2d(128, 1, kernel_size=1)
        self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

        nn.init.kaiming_normal_(self.combinedEmbedding.weight)
        nn.init.constant_(self.combinedEmbedding.bias, 0.0)

    def forward(self, v, a):
        # V video frames of 3x16x256x320
        # A audio frames of 3x16x64x64
        # return a map of 32x40

        xV1 = self.video_branch(v)
        xA1 = self.audio_branch(a)
        xA1 = xA1.expand_as(xV1)
        xC = torch.cat((xV1, xA1), dim=1)
        xC = torch.squeeze(xC, dim=2)
        x = self.combinedEmbedding(xC)
        x = F.relu(x, inplace=True)

        x = torch.squeeze(x, dim=2)
        x = self.upscale1(x)
        x = self.upscale2(x)
        sal = self.saliency(x)
        sal = F.relu(sal, inplace=True)
        return sal

