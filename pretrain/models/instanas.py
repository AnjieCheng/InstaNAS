import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import random
import torch.nn.init as torchinit
import math
from torch.nn import init, Parameter
import copy

from models import base
import utils
from utils import *


class InstaNas(nn.Module):

    def forward(self, x, policy, drop_path_prob=0):

        x = F.relu(self.bn1(self.conv1(x)))

        t = 0
        lat = Variable(torch.zeros(x.size(0)), requires_grad=False).cuda().float()
        # flops = Variable(torch.zeros(x.size(0)), requires_grad=False).cuda().float()

        for expansion, out_planes, num_blocks, stride in self.cfg:
            for idx in range(num_blocks):
                action = policy[:, t, :].contiguous()

                # early termination if all actions in the batch are zero
                if action[:, :].data.sum() == 0:
                    if idx != 0:
                        t += 1
                        continue
                    else:
                        feature_map = [self.layers[t][0](x)]
                        lat_in_this_block = [self.layers[t][0].lat.cuda() * (action[:, 0]+1)]
                        # flops_in_this_block = [self.layers[t][0].flops.cuda().float() * (action[:, 0]+1)]
                else:
                    action_mask = [action[:, i].contiguous().float().view(-1, 1, 1, 1) for i in range(action.size(1))]
                    feature_map = [self.layers[t][i](x) * action_mask[i] for i in range(action.size(1))]
                    lat_in_this_block = [self.layers[t][i].lat.cuda() * action[:, i].float() for i in range(action.size(1))]
                    # flops_in_this_block = [self.layers[t][i].flops.cuda().float() * action[:, i].float() for i in range(action.size(1))]

                x = sum(feature_map)
                lat += sum(lat_in_this_block).float()
                # flops += sum(flops_in_this_block).float()
                t += 1

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        #flops += conv1 + conv2 + linear + agent_flops

        return x, lat #, flops

    def forward_single(self, x, policy):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))

        t = 0
        for _, _, num_blocks, _ in self.cfg:
            for idx in range(num_blocks):
                feature = []
                action_mask = [policy[t, i].data.cpu().numpy() for i in range(policy.size(1))]
                if sum(action_mask) == 0:
                    # Quick skip
                    if idx == 0:
                        x = self.layers[t][0](x)
                else:
                    for i, mask in enumerate(action_mask):
                        if mask == 1:
                            feature.append(self.layers[t][i](x))
                    x = sum(feature)
                t += 1

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                block = [self._make_action(in_planes, out_planes, expansion, stride, i) for i in range(self.num_of_actions)]
                block = nn.ModuleList(block)
                layers.append(block)
                in_planes = out_planes
        self.num_of_layers = len(layers)
        print("  + Total num of layers: ", self.num_of_layers)
        return nn.Sequential(*layers)

    def _make_action(self, inp, oup, _, stride, id):
        if id == 0:  # InvertedResBlock_3x3_6F
            action = base.InvertedResBlock(inp, oup, stride, kernel=3, expansion=6)
        elif id == 1:  # InvertedResBlock_3x3_3F
            action = base.InvertedResBlock(inp, oup, stride, kernel=3, expansion=3)
        elif id == 2:  # InvertedResBlock_5x5_6F
            action = base.InvertedResBlock(inp, oup, stride, kernel=5, expansion=6)
        elif id == 3:  # InvertedResBlock_5x5_3F
            action = base.InvertedResBlock(inp, oup, stride, kernel=5, expansion=3)
        elif id == 4:  # BasicBlock
            action = base.BasicBlock(inp, oup, stride)
        else:
            raise ValueError("No such action index")
        return action

    def _profile(self, input_size):
        # Synthetic Input
        x = torch.autograd.Variable(torch.ones(1, 3, input_size, input_size)).cpu()
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))

        t = 0
        self.baseline = Variable(torch.tensor(0.), requires_grad=False) # Count mobilenetv2 syn latency
        self.baseline_max = Variable(torch.tensor(0.), requires_grad=False)
        for _, _, num_blocks, _ in self.cfg:
            for _ in range(num_blocks):
                feature = []

                for a in range(self.num_of_actions):
                    #Compute Latency of each element
                    lat = self._get_latency(self.layers[t][a], x)
                    self.layers[t][a].lat = Variable(torch.tensor(lat), requires_grad=False)
                    feature.append(self.layers[t][a](x))
                    self.baseline_max += self.layers[t][a].lat

                    if a == 0:
                        self.baseline += self.layers[t][a].lat

                x = sum(feature)
                t += 1

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

    def _get_latency(self, op, x):
        total_iter = 5
        latency_list = []
        for _ in range(total_iter):
            start = time.time()
            _ = op(x)
            latency_list.append(time.time()-start)
        return np.mean(latency_list[1:])


class MobileNet(InstaNas):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, config=None, num_classes=10):
        super(MobileNet, self).__init__()
        self.num_of_actions = 5
        self.num_of_blocks = sum([num_blocks for expansion, out_planes, num_blocks, stride in self.cfg])

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self._profile(input_size=32)


                
class MobileNet_64(InstaNas):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, config=None, num_classes=200):
        super(MobileNet_64, self).__init__()
        self.num_of_actions = 5
        self.num_of_blocks = sum([num_blocks for expansion, out_planes, num_blocks, stride in self.cfg])

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(1280, num_classes),
        )

        self._profile(input_size=64)

class MobileNet_224(InstaNas):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, config=None, num_classes=1000):
        super(MobileNet_224, self).__init__()
        self.num_of_actions = 5
        self.num_of_blocks = sum([num_blocks for expansion, out_planes, num_blocks, stride in self.cfg])

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False) # init_stride=2 for ImgNet
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        # self.linear = nn.Linear(1280, num_classes)
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes),
        )

        self._profile(input_size=224)
