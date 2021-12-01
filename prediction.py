import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from file_parser import pad_sequence

"""
PyTorch Implementation based from supplementary notes from DeepBind:
https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3300/MediaObjects/41587_2015_BFnbt3300_MOESM51_ESM.pdf
Implementation of ConvNet is implemented from the description of ConvNet in DeepBind supplementary notes

"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNet(nn.Module):
    def __init__(self, num_motifs, motif_len, max_pool, hidden_layer, training_mode, dropout_value,
                 learning_rate, learning_momentum, initial_weight,
                 neural_weight, weight_decay1, weight_decay2, weight_decay3):
        """

        :param num_motifs: int
        :param motif_len: int
        :param max_pool: bool
        :param hidden_layer: bool
        :param training_mode: bool
        :param dropout_value: float
        :param learning_rate: float
        :param learning_momentum: float
        :param initial_weight: float
        :param neural_weight: float
        """

        super(ConvNet, self).__init__()
        self.max_pool = max_pool
        self.hidden_layer = hidden_layer
        self.training_mode = training_mode
        self.dropout_value = dropout_value
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.initial_weight = initial_weight
        self.neural_weight = neural_weight
        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.weight_decay3 = weight_decay3

        self.convolution_weights = torch.randn(num_motifs, 4, motif_len).to(device)
        torch.nn.init.normal_(self.convolution_weights, std=self.initial_weight)
        self.convolution_weights.requires_grad = True

        self.rectification_weights = torch.randn(num_motifs).to(device)
        torch.nn.init.normal_(self.rectification_weights)
        self.rectification_weights = -self.rectification_weights
        self.rectification_weights.requires_grad = True

        if hidden_layer:
            if max_pool:
                self.hidden_weights = torch.randn(num_motifs, 32).to(device)
            else:
                self.hidden_weights = torch.randn(2 * num_motifs, 32).to(device)

            self.neural_weights = torch.randn(32, 1).to(device)
            self.weights_neural_bias = torch.randn(1).to(device)
            self.wHiddenBias = torch.randn(32).to(device)
            torch.nn.init.normal_(self.neural_weights, std=self.neural_weight)
            torch.nn.init.normal_(self.weights_neural_bias, std=self.neural_weight)
            torch.nn.init.normal_(self.hidden_weights, std=0.3)
            torch.nn.init.normal_(self.wHiddenBias, std=0.3)

            self.hidden_weights.requires_grad = True
            self.wHiddenBias.requires_grad = True

        else:
            if max_pool:
                self.neural_weights = torch.randn(num_motifs, 1).to(device)
            else:
                self.neural_weights = torch.randn(2 * num_motifs, 1).to(device)

            self.weights_neural_bias = torch.randn(1).to(device)
            torch.nn.init.normal_(self.neural_weights, mean=0, std=self.neural_weight)
            torch.nn.init.normal_(self.weights_neural_bias, mean=0, std=self.neural_weight)

        self.neural_weights.requires_grad = True
        self.weights_neural_bias.requires_grad = True

    def forward_pass(self, x):
        conv = F.conv1d(input=x, weight=self.convolution_weights, bias=self.rectification_weights)
        rect = torch.clamp(input=conv, min=0)
        pool, _ = torch.max(input=rect, dim=2)

        if not self.max_pool:
            avg_pool = torch.mean(input=rect, dim=2)
            pool = torch.cat(tensors=(pool, avg_pool), dim=1)

        if not self.hidden_layer:
            if self.training_mode:
                pool_drop = pool
                out = pool_drop @ self.neural_weights
                out.add_(self.weights_neural_bias)
            else:
                out = self.dropout_value * (pool @ self.neural_weights)
                out.add_(self.weights_neural_bias)

        else:
            hid = pool @ self.hidden_weights
            hid.add_(self.wHiddenBias)
            hid = hid.clamp(min=0)
            if self.training_mode:
                out = self.dropout_value * (hid @ self.neural_weights)
                out.add_(self.weights_neural_bias)
            else:
                out = self.dropout_value * (hid @ self.neural_weights)
                out.add_(self.weights_neural_bias)

        return out

    def forward(self, x):
        out = self.forward_pass(x)
        return out

    def predict(self, seq):
        padded_seq = pad_sequence(sequence=seq, motif_len=24, kind='DNA')
        padded_seq = np.expand_dims(padded_seq, axis=0)
        x = torch.from_numpy(padded_seq)
        out = self.forward_pass(x)
        return out.item()
