import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.function import once_differentiable
from ops import _C

__all__ = ['LocalGridAlignAttention', 'ga_align_weight', 'ga_align_map', 'GaAlignModule']


class _GaAlignWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key):
        weight = _C.ga_align_forward(query, key)

        ctx.save_for_backward(query, key)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key = ctx.saved_tensors

        dq, dk = _C.ga_align_backward(dw, query, key)
        return dq, dk


class _GaAlignMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, value):
        out = _C.ga_map_align_forward(weight, value)

        ctx.save_for_backward(weight, value)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, value = ctx.saved_tensors

        dw, dv = _C.ga_map_backward(dout, weight, value)

        return dw, dv


ga_align_weight = _GaAlignWeight.apply
ga_align_map = _GaAlignMap.apply


class LocalGridAlignAttention(nn.Module):
    """Local Grid Align Attention Module"""

    def __init__(self, in_channels):
        super(LocalGridAlignAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.scaling = float(in_channels // 8) ** -0.5

    def forward(self, x, featmap):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(featmap)
        proj_query = proj_query * self.scaling

        bsz, embed_dim, h, w = proj_query.size()
        hf, wf = proj_key.shape[2:]
        head_dim = embed_dim // 8
        assert head_dim * 8 == embed_dim, "embed_dim must be divisible by num_heads"
        proj_query = proj_query.contiguous().view(bsz * 8, head_dim, h, w)
        proj_key = proj_key.contiguous().view(bsz * 8, head_dim, hf, wf)
        featmap = featmap.contiguous().view(bsz * 8, head_dim, hf, wf)

        energy = ga_align_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ga_align_map(attention, featmap)
        out = out.contiguous().view(bsz, embed_dim, h, w)

        return out


class GaAlignModule(nn.Module):
    def __init__(self, in_channels):
        super(GaAlignModule, self).__init__()
        self.ga_align = LocalGridAlignAttention(in_channels)
        self.convb = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, featmap):
        out = self.ga_align(x, featmap)
        out = self.convb(out)

        return out
