#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YijieDang
"""

import torch
import torch.nn as nn
from itertools import repeat

from typing import Union, Sequence
from . import capsule_layer as cap
# import capsule_layer as cap
from monai.networks.blocks.transformerblock import TransformerBlock

device = torch.device("cuda:0")

# Attention Block
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
##############################################################################
# ResBlock   
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x) + self.skip(x)
##############################################################################
# D-TrAttUnet Architecture
class CADUnet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            routing=3):
        super().__init__()

        self.classification = False
        self.CapsuleRouting=routing
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        self.capconv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)

        nb_filter = [32, 64, 128, 256, 512]  # 0~4层的特征图个数
        self.encoder1 = DoubleConv(in_channels, nb_filter[0])

        self.conv1 = DoubleConv(in_channels, nb_filter[0])
        self.conv2 = DoubleConv(nb_filter[1], nb_filter[1])
        self.conv3 = DoubleConv(nb_filter[2], nb_filter[2])
        self.conv4 = DoubleConv(nb_filter[3], nb_filter[3])
        self.conv5 = DoubleConv(nb_filter[4], nb_filter[4])
        # Define the capsule convolutional layer.
        self.cap1 = cap.CapsuleLayer(t_0=1, z_0=16, t_1=2, z_1=16, s=2,routing=1)
        self.cap2 = cap.CapsuleLayer(t_0=2, z_0=16, t_1=2, z_1=32, s=2, routing=self.CapsuleRouting)
        self.cap3 = cap.CapsuleLayer(t_0=2, z_0=32, t_1=4, z_1=32, s=2, routing=self.CapsuleRouting)
        self.cap4 = cap.CapsuleLayer(t_0=4, z_0=32, t_1=4, z_1=64, s=2, routing=self.CapsuleRouting)

        self.Att4 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3])
        self.Att3 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Att2 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Att1 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0])

        self.deconv1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.deconv2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.deconv3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.deconv4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.outconv1_d1 = nn.Conv2d(in_channels=nb_filter[3], out_channels=out_channels,
                                     kernel_size=1)  # Decoder1输出层从下往上第一层
        self.outconv2_d1 = nn.Conv2d(in_channels=nb_filter[2], out_channels=out_channels, kernel_size=1)  # 输出层从下往上第二层
        self.outconv3_d1 = nn.Conv2d(in_channels=nb_filter[1], out_channels=out_channels, kernel_size=1)  # …………第三层
        self.outconv4_d1 = nn.Conv2d(in_channels=nb_filter[0], out_channels=out_channels, kernel_size=1)  # …………第四层

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

        # Decoder2
        self.Att41 = Attention_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3])
        self.Att31 = Attention_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Att21 = Attention_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Att11 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0])

        self.deconv11 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.deconv21 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.deconv31 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.deconv41 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size, feat_size, hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in):
        B,H,W,C=x_in.shape
        # Encoder
        x1 = self.conv1(x_in)
        x_cap = self.capconv(x_in)
        x_cap = torch.unsqueeze(x_cap, dim=1)
        mp_x1=self.pool(x1)
        x_cap1 = self.cap1(x_cap)
        p1 = torch.flatten(x_cap1, start_dim=1, end_dim=2)
        pout1=p1*mp_x1
        pin2=torch.reshape(pout1,(B,2,16,112,112))
        x2 = self.conv2(torch.cat([mp_x1, pout1], dim=1))

        mp_x2=self.pool(x2)
        x_cap2 = self.cap2(pin2)
        p2 = torch.flatten(x_cap2, start_dim=1, end_dim=2)
        pout2=p2*mp_x2
        pin3=torch.reshape(pout2,(B,2,32,56,56))
        x3 = self.conv3(torch.cat([mp_x2, pout2], dim=1))

        mp_x3=self.pool(x3)
        x_cap3 = self.cap3(pin3)
        p3 = torch.flatten(x_cap3, start_dim=1, end_dim=2)
        pout3=p3*mp_x3
        pin4=torch.reshape(pout3,(B,4,32,28,28))
        x4 = self.conv4(torch.cat([mp_x3, pout3], dim=1))

        mp_x4=self.pool(x4)
        x_cap4 = self.cap4(pin4)
        p4 = torch.flatten(x_cap4, start_dim=1, end_dim=2)
        pout4=p4*mp_x4
        x5 = self.conv5(torch.cat([mp_x4, pout4], dim=1))

        # Encoder
        # 1
        x50 = self.up(x5)
        xd4 = self.Att4(g=x50, x=x4)
        d1= self.deconv1(torch.cat([xd4, x50], 1))
        # 2
        x51 = self.up(x5)
        xd41 = self.Att41(g=x51, x=x4)
        d11 = self.deconv11(torch.cat([xd41, x51], 1))

        # 1
        x40 = self.up(d1)
        xd3 = self.Att3(g=x40, x=x3)
        d2 = self.deconv2(torch.cat([xd3, x40], 1))
        # 2
        x41 = self.up(d11)
        xd31 = self.Att31(g=x41, x=x3)
        d21 = self.deconv21(torch.cat([xd31, x41], 1))

        # 1
        x30 = self.up(d2)
        xd2 = self.Att2(g=x30, x=x2)
        d3 = self.deconv3(torch.cat([xd2, x30], 1))
        # 2
        x31 = self.up(d21)
        xd21 = self.Att21(g=x31, x=x2)
        d31 = self.deconv31(torch.cat([xd21, x31], 1))

        # 1
        x20 = self.up(d3)
        xd1 = self.Att1(g=x20, x=x1)
        d4 = self.deconv4(torch.cat([xd1, x20], 1))
        # 2
        x21 = self.up(d31)
        xd11 = self.Att11(g=x21, x=x1)
        d41 = self.deconv41(torch.cat([xd11, x21], 1))

        output=self.final(d4)
        output2=self.final1(d41)
        return output, output2