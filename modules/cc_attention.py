import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.function import once_differentiable
from ops import _C

__all__ = ['CrissCrossAttention', 'ca_weight', 'ca_map']


class _CAWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        weight = _C.ca_forward(t, f)

        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt, df = _C.ca_backward(dw, t, f)
        return dt, df


class _CAMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        out = _C.ca_map_forward(weight, g)

        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw, dg = _C.ca_map_backward(dout, weight, g)

        return dw, dg


ca_weight = _CAWeight.apply
ca_map = _CAMap.apply


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_channels, bias=True):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x

        return out


class RCCAModule(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, recurrence=2):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            #norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 3, padding=1, bias=False),
            #norm_layer(in_channels),
            nn.ReLU(True))

    def forward(self, x):
        out = self.conva(x)
        for i in range(self.recurrence):
            out = self.cca(out)
        out = self.convb(out)

        return out

#class MultiHeadCCAttention(nn.Module):
    """Multi Head Criss-Cross Attention Module"""

    """def __init__(self, in_channels, num_heads, bias=True):
        super(MultiHeadCCAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)"""
