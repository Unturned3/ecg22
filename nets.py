
from argparse import ArgumentError
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Swish activation function
# Original paper: https://arxiv.org/pdf/1710.05941v1
# Used by Wickramasinghe et al. for ECG tasks (ref. [17] in survey.pdf)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def swish_visualize():
    x = torch.linspace(-10.0, 10.0, 100)
    swish = Swish()
    swish_out = swish(x)
    relu_out = torch.relu(x)

    plt.title('Swish function')
    plt.plot(x.numpy(), swish_out.numpy(), label='Swish')
    plt.plot(x.numpy(), relu_out.numpy(), label='ReLU')
    plt.legend()
    plt.show()


# This block contains conv, norm, and pooling
class Conv_Block(nn.Module):

    def __init__(self, input_size, hidden_size,
                 kernel_size, norm_type='batchnorm'):

        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'batchnorm':
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
        else:
            raise ArgumentError('Only batchnorm is supported!')

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        #print("x.shape before pad: {}".format(x.shape))
        # pad to keep dim constant
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        #print("x.shape after pad: {}".format(x.shape))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size=1, hid_size=256,
                kernel_size=5, num_classes=5):

        super().__init__()

        self.conv1 = Conv_Block(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = Conv_Block(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = Conv_Block(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        #self.pool = nn.MaxPool1d((1))
        #self.avgpool = nn.AvgPool1d((1))
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size//4, out_features=num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        # DEBUG
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x


class RNN_Block(nn.Module):

    def __init__(self, input_size, hid_size, num_rnn_layers=1,
                 dropout_p=0.2, bidirectional=False, rnn_type='lstm'):

        super().__init__()

        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states


class RNNModel(nn.Module):
    def __init__(self, input_size, hid_size, rnn_type,
                 bidirectional, n_classes=5, kernel_size=5):

        super().__init__()

        self.rnn_layer = RNN_Block(
            input_size=46,  # hid_size * 2 if bidirectional else hid_size,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = Conv_Block(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = Conv_Block(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x, _ = self.rnn_layer(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)  # .squeeze(1)
        return x
