import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, constant_
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
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.grid_conv = nn.Conv2d(in_channels, 1, 1)
        self.scaling = float(in_channels // 8) ** -0.5

        xavier_uniform_(self.query_conv.weight.data)
        constant_(self.query_conv.bias.data, 0.)
        xavier_uniform_(self.key_conv.weight.data)
        constant_(self.key_conv.bias.data, 0.)
        xavier_uniform_(self.value_conv.weight.data)
        constant_(self.value_conv.bias.data, 0.)
        constant_(self.grid_conv.weight.data, 0.)
        constant_(self.grid_conv.bias.data, 0.)

    def forward(self, x, featmap, padding_mask=None):
        grid_weight = self.grid_conv(x).sigmoid()  # (b, 1, 10, 10)
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(featmap)
        proj_value = self.value_conv(featmap)
        if padding_mask is not None:
            proj_value = proj_value.masked_fill(padding_mask[:, None], float(0))
        proj_query = proj_query * self.scaling

        bsz, embed_dim, h, w = proj_query.size()
        hf, wf = proj_key.shape[2:]
        head_dim = embed_dim // 8
        assert head_dim * 8 == embed_dim, "embed_dim must be divisible by num_heads"
        proj_query = proj_query.contiguous().view(bsz * 8, head_dim, h, w)
        proj_key = proj_key.contiguous().view(bsz * 8, head_dim, hf, wf)
        proj_value = proj_value.contiguous().view(bsz * 8, head_dim, hf, wf)

        attn_weight = ga_align_weight(proj_query, proj_key)
        attention = F.softmax(attn_weight, 1)
        out = ga_align_map(attention, proj_value)  # (b * 8, c, 10, 10)
        out = out.contiguous().view(bsz, embed_dim, h, w) * grid_weight

        return out


class GaAlignModule(nn.Module):
    def __init__(self, in_channels):
        super(GaAlignModule, self).__init__()
        self.ga_align = LocalGridAlignAttention(in_channels)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

        xavier_uniform_(self.out_conv.weight.data)
        constant_(self.out_conv.bias.data, 0.)

    def forward(self, x, featmap, padding_mask=None):
        out = self.ga_align(x, featmap, padding_mask)
        out = self.out_conv(out)

        return out
