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

    def __init__(self, in_channels, num_heads=8, bias=True):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        """ Multi-head CC """
        bsz, embed_dim, h, w = proj_query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        proj_query = proj_query.contiguous().view(bsz * self.num_heads, head_dim, h, w)
        proj_key = proj_key.contiguous().view(bsz * self.num_heads, head_dim, h, w)
        proj_value = proj_value.contiguous().view(bsz * self.num_heads, head_dim, h, w)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = out.contiguous().view(bsz, embed_dim, h, w)
        out = self.gamma * out + x

        return out


class CrissCrossAttentionDecoder(nn.Module):
    """Criss-Cross Attention Decoder Module"""

    def __init__(self, in_channels):
        super(CrissCrossAttentionDecoder, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.scaling = float(in_channels // 8) ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query, memory):
        proj_query = self.query_conv(query)
        proj_key = self.key_conv(memory)
        proj_value = self.value_conv(memory)
        proj_query = proj_query * self.scaling

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + memory

        return out


class RCCAModule(nn.Module):
    def __init__(self, in_channels, recurrence=2):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(in_channels)

        self.bottleneck = nn.Sequential(
        #    norm_layer(in_channels),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=True),
            nn.ReLU(True),
        #    nn.Dropout2d(0.1),
        )
        #self.norm1 = norm_layer(32, inter_channels)
        #self.norm2 = norm_layer(32, inter_channels)
        #self.dropout1 = nn.Dropout2d(0.1)
        #self.dropout2 = nn.Dropout2d(0.1)

    def forward(self, x):
        out = x.clone()
        for i in range(self.recurrence):
            out = self.cca(out)
        #out = self.norm1(out)
        #out = self.convb(self.conva(out))
        #out = self.convb(out)
        #out = out + out2
        #out = self.norm2(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out


class RCCAModuleNormal(nn.Module):
    def __init__(self, in_channels, recurrence=2):
        super(RCCAModuleNormal, self).__init__()
        self.recurrence = recurrence
        self.cca = CrissCrossAttentionDecoder(in_channels)

        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, query, memory):
        memory = self.cca(query, memory)
        memory = self.cca(query, memory)
        out = self.out_conv(memory)
        return out
