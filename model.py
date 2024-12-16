# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 19:56
# @Author  : lil louis
# @Location: Beijing
# @File    : model.py

import torch.nn as nn
import torch


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_down_sample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_down_sample = identity_down_sample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_down_sample is not None:
            identity = self.identity_down_sample(identity)
        x += identity
        x = self.relu(x)
        return x


class CO_Single(nn.Module):
    def __init__(self, in_channels, block, layers):
        super(CO_Single, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_features = 2048 + 4 + 7
        self.in_channels = 64
        self.init_conv = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.all_features),
            nn.Linear(in_features=self.all_features, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_down_sample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4))
        layers.append(block(self.in_channels, out_channels, identity_down_sample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_block - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, img_dim, physio_dim, oral_dim):
        img_dim = self.init_conv(img_dim)
        img_dim = self.init_bn(img_dim)
        img_dim = self.relu(img_dim)
        img_dim = self.maxpool(img_dim)
        img_dim = self.layer1(img_dim)
        img_dim = self.layer2(img_dim)
        img_dim = self.layer3(img_dim)
        img_dim = self.layer4(img_dim)
        img_dim = self.avg_pool(img_dim)
        img_dim = img_dim.squeeze(-1).squeeze(-1)

        all_dim = torch.cat([img_dim, physio_dim, oral_dim], dim=1)
        all_dim = torch.sigmoid(all_dim)
        result = self.MLP(all_dim)
        return result


class CO_Both(nn.Module):
    def __init__(self, in_channels, block, layers):
        super(CO_Both, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_features = 2048 + 2048 + 4 + 7
        self.in_channels = 64
        self.init_conv = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.all_features),
            nn.Linear(in_features=self.all_features, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_down_sample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4))
        layers.append(block(self.in_channels, out_channels, identity_down_sample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_block - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, img_dim_1, img_dim_2, physio_dim, oral_dim):
        img_dim_1 = self.init_conv(img_dim_1)
        img_dim_1 = self.init_bn(img_dim_1)
        img_dim_1 = self.relu(img_dim_1)
        img_dim_1 = self.maxpool(img_dim_1)
        img_dim_1 = self.layer1(img_dim_1)
        img_dim_1 = self.layer2(img_dim_1)
        img_dim_1 = self.layer3(img_dim_1)
        img_dim_1 = self.layer4(img_dim_1)
        img_dim_1 = self.avg_pool(img_dim_1)
        img_dim_1 = img_dim_1.squeeze(-1).squeeze(-1)

        img_dim_2 = self.init_conv(img_dim_2)
        img_dim_2 = self.init_bn(img_dim_2)
        img_dim_2 = self.relu(img_dim_2)
        img_dim_2 = self.maxpool(img_dim_2)
        img_dim_2 = self.layer1(img_dim_2)
        img_dim_2 = self.layer2(img_dim_2)
        img_dim_2 = self.layer3(img_dim_2)
        img_dim_2 = self.layer4(img_dim_2)
        img_dim_2 = self.avg_pool(img_dim_2)
        img_dim_2 = img_dim_2.squeeze(-1).squeeze(-1)

        all_dim = torch.cat([img_dim_1, img_dim_2, physio_dim, oral_dim], dim=1)
        all_dim = torch.sigmoid(all_dim)
        result = self.MLP(all_dim)
        return result