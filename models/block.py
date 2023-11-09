from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MergeAndConv(nn.Module):

    def __init__(self, ic, oc, inner=32):
        super().__init__()

        self.conv1 = nn.Conv2d(ic, inner, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(inner)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inner, oc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2(self.bn(self.relu(self.conv1(x))))
        x = torch.sigmoid(x)
        return x


class SideClassifer(nn.Module):
    def __init__(self, ic, n_class=1, M=2, kernel_size=1):
        super().__init__()

        sides = []
        for i in range(M):
            sides.append(nn.Conv2d(ic, n_class, kernel_size=kernel_size))

        self.sides = nn.ModuleList(sides)

    def forward(self, x):
        return [fn(x) for fn in self.sides]


class UpsampleSKConv(nn.Module):
    """docstring for UpsampleSKConvPlus"""

    def __init__(self, ic, oc, reduce=4):
        super(UpsampleSKConv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.prev = nn.Conv2d(ic, ic // reduce, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ic // reduce)

        self.next = nn.Conv2d(ic // reduce, oc, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(oc)

        self.sk = SKSPP(ic // reduce, ic // reduce, M=4)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)

        x = self.bn(self.relu(self.prev(x)))

        x = self.sk(x)

        x = self.bn2(self.relu(self.next(x)))

        return x


class SKSPP(nn.Module):
    def __init__(self, features, WH, M=2, G=1, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKSPP, self).__init__()
        d = max(int(features / r), L)
        self.M = M  # original
        self.features = features
        self.convs = nn.ModuleList([])

        # 1,3,5,7 padding:[0,1,2,3]
        for i in range(1, M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=1 + i * 2, dilation=1 + i * 2, stride=stride,
                          padding=((1 + i * 2) * (i * 2) + 1) // 2, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        feas = torch.unsqueeze(x, dim=1)

        # F->conv1x1->conv3x3->conv5x5->conv7x7

        for i, conv in enumerate(self.convs):
            x = conv(x)
            # if i == 0:
            #     feas = fea
            # else:
            feas = torch.cat([feas, torch.unsqueeze(x, dim=1)], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v