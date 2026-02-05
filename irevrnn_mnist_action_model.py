# -*-coding:gbk-*-

from typing import Optional
import torch
import torchvision
from torch import nn, Tensor
from model.torch_irevrnn.irevrnn import IRevRNN
from model.torch_irevrnn.common import inddropout, SeqWrap
import time


class MnistActionIRevRNNPlain(nn.Module):
    def __init__(self, num_layers: int, input_size: int, output_size: int, hidden_size: int,
                 rev_len: int, ind_act_typ_str: str, res_act_typ_str: str):
        super(MnistActionIRevRNNPlain, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList()
        self.irevrnn = nn.ModuleList()
        self.bn = nn.ModuleList()
        first_linear_layer = nn.Linear(input_size, hidden_size, bias=True)
        wrapped_linear_layer = SeqWrap(first_linear_layer)
        self.linear.append(wrapped_linear_layer)
        for i in range(num_layers):
            if i != 0:
                linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)
                wrapped_linear_layer = SeqWrap(linear_layer)
                self.linear.append(wrapped_linear_layer)
            irevrnn_layer = IRevRNN(hidden_size, rev_len, ind_act_typ_str, res_act_typ_str).cuda()
            self.irevrnn.append(irevrnn_layer)
            bn_layer = nn.BatchNorm1d(hidden_size)
            wrapped_bn_layer = SeqWrap(bn_layer)
            self.bn.append(wrapped_bn_layer)
        self.classifier = nn.Linear(hidden_size, output_size, bias=True)
        self.reset_parameters()

    # this function is from indRNN
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'irevrnn.' + str(self.num_layers - 1) + '.ind_weights' in name:
                param.data.uniform_(0.9, 1)
            if 'linear' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bn' in name and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, input):
        output = input
        for i in range(self.num_layers):
            output = self.linear[i](output)
            output = self.irevrnn[i](output)
            output = self.bn[i](output)
            output = inddropout(output, self.training, p=0.1)
        output = self.classifier(output[-1])
        return output


class _ResidualLayer(nn.Sequential):
    def __init__(self, hidden_size, rev_len, ind_act_typ_str, res_act_typ_str):
        super(_ResidualLayer, self).__init__()
        linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear1 = SeqWrap(linear_layer)
        bn_layer = nn.BatchNorm1d(hidden_size)
        self.bn1 = SeqWrap(bn_layer)
        self.irevrnn1 = IRevRNN(hidden_size, rev_len, ind_act_typ_str, res_act_typ_str).cuda()
        self.irevrnn2 = IRevRNN(hidden_size, rev_len, ind_act_typ_str, res_act_typ_str).cuda()
        linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear2 = SeqWrap(linear_layer)
        bn_layer = nn.BatchNorm1d(hidden_size)
        self.bn2 = SeqWrap(bn_layer)

    def forward(self, input):
        output = input
        output = self.irevrnn1(output)
        output = self.bn1(output)
        output = inddropout(output, self.training, p=0.1)
        output = self.linear1(output)
        output = self.irevrnn2(output)
        output = self.bn2(output)
        output = inddropout(output, self.training, p=0.1)
        output = self.linear2(output)
        output += input
        return output


class MnistActionIRevRNNResNet(nn.Module):
    def __init__(self, num_layers: int, input_size: int, output_size: int, hidden_size: int,
                 rev_len: int, ind_act_typ_str: str, res_act_typ_str: str):
        super(MnistActionIRevRNNResNet, self).__init__()
        self.num_layers = num_layers
        self.res = nn.ModuleList()
        linear_layer = nn.Linear(input_size, hidden_size, bias=True)
        wrapped_linear_layer = SeqWrap(linear_layer)
        self.add_module('linear', wrapped_linear_layer)
        for i in range(num_layers):
            res_layer = _ResidualLayer(hidden_size, rev_len, ind_act_typ_str, res_act_typ_str)
            self.res.append(res_layer)
        irevrnn_layer = IRevRNN(hidden_size, rev_len, ind_act_typ_str, res_act_typ_str).cuda()
        self.add_module('irevrnn', irevrnn_layer)
        bn_layer = nn.BatchNorm1d(hidden_size)
        wrapped_bn_layer = SeqWrap(bn_layer)
        self.add_module('bn', wrapped_bn_layer)

        self.classifier = nn.Linear(hidden_size, output_size, bias=True)
        self.reset_parameters()

    # this function is from indRNN
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'irevrnn.' + str(self.num_layers - 1) + '.ind_weights' in name:
                param.data.uniform_(0.9, 1)
            if 'linear' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bn' in name and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, input):
        output = input
        output = self.linear(output)
        for i in range(self.num_layers):
            output = self.res[i](output)

        output = self.irevrnn(output)
        output = self.bn(output)
        output = inddropout(output, self.training, p=0.1)

        output = self.classifier(output[-1])
        return output
