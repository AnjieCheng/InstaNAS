import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy

from models import base
from utils import *

class ResNet(nn.Module):

    def seed(self, x):
        raise NotImplementedError

    def forward(self, x, policy):

        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                action = policy[:, t].contiguous()
                residual = self.ds[segment](x) if b == 0 else x

                if action.data.sum() == 0:
                    x = residual
                    t += 1
                    continue

                action_mask = action.float().view(-1, 1, 1, 1)
                fx = F.relu(residual + self.blocks[segment][b](x))
                x = fx*action_mask + residual*(1-action_mask)
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_single(self, x, policy):
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                if policy[t] == 1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_full(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet32(ResNet):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet32, self).__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for _, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(
                block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = base.DownsampleB(
                self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


class ResNet224(ResNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64//4
        super(ResNet224, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64//4, 128//4, 256//4, 512//4]
        self.blocks, self.ds = [], []
        for _, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(
                block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion //4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.layer_config = layers

    def seed(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers, downsample


class Policy32(nn.Module):

    def __init__(self, layer_config=[1, 1, 1], num_blocks=15, num_of_actions=7):
        super(Policy32, self).__init__()
        self.num_of_actions = num_of_actions  # 7
        self.num_blocks = num_blocks
        self.features = ResNet32(base.BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()
        self.logit = nn.Linear(self.feat_dim, num_blocks * self.num_of_actions)

    def load_state_dict(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)

    def forward(self, x):
        x = self.features.forward_full(x)
        probs = torch.sigmoid(self.logit(x)).view(-1, self.num_blocks, self.num_of_actions)
        return probs, None


class Policy224(nn.Module):

    def __init__(self, layer_config=[1, 1, 1, 1], num_blocks=17, num_of_actions=5):
        super(Policy224, self).__init__()
        self.num_blocks = num_blocks
        self.num_of_actions = num_of_actions
        self.features = ResNet224(base.BasicBlock, layer_config, num_classes=1000)

        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks * self.num_of_actions)

    def load_state_dict(self, state_dict):
        new_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k.startswith('features.fc'):
                continue
            new_state_dict[k] = v
        return super(Policy224, self).load_state_dict(new_state_dict)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.features.forward_full(x)
        probs = torch.sigmoid(self.logit(x)).view(-1, self.num_blocks, self.num_of_actions)
        return probs, None

