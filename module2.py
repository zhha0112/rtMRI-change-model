import copy
import math
from collections import OrderedDict

import hyperparams as hp
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True,
                 w_init='linear'):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    def __init__(self, embedding_size, num_hidden):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv(embedding_size, num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.conv2 = Conv(num_hidden, num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.conv3 = Conv(num_hidden, num_hidden, kernel_size=5, padding=int(np.floor(5 / 2)), w_init='relu')
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = input_.transpose(1, 2)
        input_ = self.dropout1(t.relu(self.batch_norm1(self.conv1(input_))))
        input_ = self.dropout2(t.relu(self.batch_norm2(self.conv2(input_))))
        input_ = self.dropout3(t.relu(self.batch_norm3(self.conv3(input_))))
        input_ = input_.transpose(1, 2)
        input_ = self.projection(input_)
        return input_


class FFN(nn.Module):
    def __init__(self, num_hidden):
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        x = input_.transpose(1, 2)
        x = self.w_2(t.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        x = x + input_
        x = self.layer_norm(x)
        return x


class PostConvNet(nn.Module):
    def __init__(self, num_hidden):
        super(PostConvNet, self).__init__()
        self.upsample1 = nn.ConvTranspose1d(num_hidden, num_hidden, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(num_hidden, num_hidden, kernel_size=4, stride=2, padding=1)
        self.conv_final = nn.Conv1d(num_hidden, hp.num_mels, kernel_size=1)

    def forward(self, input_):
        input_ = F.relu(self.upsample1(input_))
        input_ = F.relu(self.upsample2(input_))
        input_ = self.conv_final(input_)
        return input_


class MultiheadAttention(nn.Module):
    def __init__(self, num_hidden_k):
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        attn = t.bmm(query, key.transpose(1, 2)) / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
        attn = t.softmax(attn, dim=-1)
        if query_mask is not None:
            attn = attn * query_mask
        result = t.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    def __init__(self, num_hidden, h=4):
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        batch_size = memory.size(0)
        seq_k, seq_q = memory.size(1), decoder_input.size(1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k).repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)

        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn).permute(2, 0, 1,
                                                                                                 3).contiguous().view(
            -1, seq_k, self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn).permute(2, 0, 1,
                                                                                                     3).contiguous().view(
            -1, seq_k, self.num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn).permute(2, 0, 1,
                                                                                                            3).contiguous().view(
            -1, seq_q, self.num_hidden_per_attn)

        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn).permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)
        result = self.layer_norm(result + decoder_input)
        return result, attns


class Prenet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        super(Prenet, self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(input_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p)),
            ('fc2', Linear(hidden_size, output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):
        return self.layer(input_)


class CBHG(nn.Module):
    def __init__(self, hidden_size, K=16, projection_size=512, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList(
            [nn.Conv1d(projection_size, hidden_size, kernel_size=i, padding=int(np.floor(i / 2))) for i in
             range(1, K + 1)])
        self.batchnorm_list = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(K)])
        convbank_outdim = hidden_size * K
        self.conv_projection_1 = nn.Conv1d(convbank_outdim, hidden_size, kernel_size=3, padding=1)
        self.conv_projection_2 = nn.Conv1d(hidden_size, projection_size, kernel_size=3, padding=1)
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers, batch_first=True,
                          bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        return x[:, :, :-1] if kernel_size % 2 == 0 else x

    def forward(self, input_):
        convbank_list = [self._conv_fit_dim(t.relu(batchnorm(conv(input_))), k + 1) for conv, batchnorm, k in
                         zip(self.convbank_list, self.batchnorm_list, range(len(self.convbank_list)))]
        conv_cat = t.cat(convbank_list, dim=1)
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]
        conv_projection = self.batchnorm_proj_1(t.relu(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_
        highway = self.highway.forward(conv_projection.transpose(1, 2))
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)
        return out


class Highwaynet(nn.Module):
    def __init__(self, num_units, num_layers=4):
        super(Highwaynet, self).__init__()
        self.linears = nn.ModuleList([Linear(num_units, num_units) for _ in range(num_layers)])
        self.gates = nn.ModuleList([Linear(num_units, num_units) for _ in range(num_layers)])

    def forward(self, input_):
        out = input_
        for fc1, fc2 in zip(self.linears, self.gates):
            h = t.relu(fc1(out))
            t_ = t.sigmoid(fc2(out))
            out = h * t_ + out * (1 - t_)
        return out
