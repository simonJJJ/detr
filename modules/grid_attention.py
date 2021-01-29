import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.function import once_differentiable
from ops import _C

__all__ = ['LocalGirdAttention', 'ga_weight', 'ga_map']


class _GAWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key):
        weight = _C.ga_forward(query, key)

        ctx.save_for_backward(query, key)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key = ctx.saved_tensors

        dq, dk = _C.ga_backward(dw, query, key)
        return dq, dk


class _GAMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, value):
        out = _C.ga_map_forward(weight, value)

        ctx.save_for_backward(weight, value)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, value = ctx.saved_tensors

        dw, dv = _C.ga_map_backward(dout, weight, value)

        return dw, dv


ga_weight = _GAWeight.apply
ga_map = _GAMap.apply


class LocalGirdAttention(nn.Module):
    """Local Grid Attention Module"""

    def __init__(self, in_channels):
        super(LocalGirdAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        #self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.scaling = float(in_channels // 8) ** -0.5

    def forward(self, x, featmap):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(featmap)
        #proj_value = self.value_conv(featmap)
        proj_query = proj_query * self.scaling

        bsz, embed_dim, h, w = proj_query.size()
        hf, wf = proj_key.shape[2:]
        head_dim = embed_dim // 8
        assert head_dim * 8 == embed_dim, "embed_dim must be divisible by num_heads"
        proj_query = proj_query.contiguous().view(bsz * 8, head_dim, h, w)
        proj_key = proj_key.contiguous().view(bsz * 8, head_dim, hf, wf)
        featmap = featmap.contiguous().view(bsz * 8, head_dim, hf, wf)

        energy = ga_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ga_map(attention, featmap)
        out = out.contiguous().view(bsz, embed_dim, h, w)

        return out


class GAModule(nn.Module):
    def __init__(self, in_channels):
        super(GAModule, self).__init__()
        self.ga = LocalGirdAttention(in_channels)
        self.convb = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, featmap):
        out = self.ga(x, featmap)
        out = self.convb(out)

        return out
