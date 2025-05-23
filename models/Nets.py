#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                          #[1,28,28] -> 
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, args.num_classes)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifar_denser(nn.Module):
    def __init__(self, args):
        super(CNNCifar_denser, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.bn5(x)
        x = self.pool(F.relu(self.conv6(x)))
        x = self.bn6(x)
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.bn(self.conv1(x)))


        return F.log_softmax(x, dim=1)
    
class SimpleConvNet(nn.Module):
    def __init__(self, args, in_channels=3, num_classes=10, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for conv in self.layers:
            x = self.gn_relu(conv(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNNFemnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)


class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        # self.h0 = torch.zeros(2, batch_size, 256).requires_grad_()
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 80)

    def forward(self, x):
        x = self.embed(x)
        # if self.h0.size(1) == x.size(0):
        #     self.h0.data.zero_()
        #     # self.c0.data.zero_()
        # else:
        #     # resize hidden vars
        #     device = next(self.parameters()).device
        #     self.h0 = torch.zeros(2, x.size(0), 256).to(device).requires_grad_()
        x, hidden = self.lstm(x)
        x = self.drop(x)
        # x = x.contiguous().view(-1, 256)
        # x = x.contiguous().view(-1, 256)
        return self.out(x[:, -1, :])

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #
    #     initial_hidden = (weight.new(2, batch_size, 256).zero_(),
    #                       weight.new(2, batch_size, 256).zero_())
    #
    #     return initial_hidden




class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1*1*512, 512),       # nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 10),         # nn.Linear(4096, 4096),
            nn.ReLU())
        # self.fc3= nn.Sequential(
        #     nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)            # x=[128, 3, 32, 32]
        out = self.layer2(out)          # [64, 32, 32]
        out = self.layer3(out)          # [64, 16, 16]
        out = self.layer4(out)          # [128, 16, 16]
        out = self.layer5(out)          # [128, 8, 8]
        out = self.layer6(out)          # [256, 8, 8]
        out = self.layer7(out)          # [256, 8, 8]
        out = self.layer8(out)          # [256, 4, 4]
        out = self.layer9(out)          # [512, 4, 4]
        out = self.layer10(out)         # [512, 4, 4]
        out = self.layer11(out)         # [512, 2, 2]
        out = self.layer12(out)         # [512, 2, 2]
        out = self.layer13(out)         # [512, 2, 2]
        out = out.reshape(out.size(0), -1)  # [512, 1, 1]  -  [512]
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        return out
    


