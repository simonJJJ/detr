import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, constant_
from torch.autograd.function import once_differentiable
from ops import _C

__all__ = ['CrossCCAttention', 'cca_map']


class _CCAWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, point):
        out = _C.cross_ca_weight_forward(query, key, point)

        ctx.save_for_backward(query, key, point)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key, point = ctx.saved_tensors

        grad_query, grad_key, grad_point = _C.cross_ca_weight_backward(dw, query, key, point)

        return grad_query, grad_key, grad_point


class _CCAMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, value, point):
        out = _C.cross_ca_map_forward(weight, value, point)

        ctx.save_for_backward(weight, value, point)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, value, point = ctx.saved_tensors

        grad_value, grad_weight, grad_point = _C.cross_ca_map_backward(dout, weight, value, point)

        return grad_weight, grad_value, grad_point


cca_weight = _CCAWeight.apply
cca_map = _CCAMap.apply


class CrossCCAttention(nn.Module):
    """Cross Criss-Cross Attention Module"""

    def __init__(self, in_channels, num_heads=8):
        super(CrossCCAttention, self).__init__()
        self.query_conv = nn.Linear(in_channels, in_channels)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.output_conv = nn.Linear(in_channels, in_channels)
        self.scaling = float(in_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        xavier_uniform_(self.query_conv.weight.data)
        constant_(self.query_conv.bias.data, 0.)
        xavier_uniform_(self.key_conv.weight.data)
        constant_(self.key_conv.bias.data, 0.)
        xavier_uniform_(self.value_conv.weight.data)
        constant_(self.value_conv.bias.data, 0.)
        xavier_uniform_(self.output_conv.weight.data)
        constant_(self.output_conv.bias.data, 0.)

    def forward(self, query, key, value, sampling_points, padding_mask=None):
        proj_query = self.query_conv(query)
        proj_key = self.key_conv(key)
        proj_value = self.value_conv(value)
        if padding_mask is not None:
            proj_value = proj_value.masked_fill(padding_mask[:, None], float(0))
        proj_query = proj_query * self.scaling
        bsz, embed_dim, h, w = proj_value.shape
        num_query = query.shape[1]
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        proj_query = proj_query.contiguous().reshape(bsz, num_query, self.num_heads, head_dim)
        proj_key = proj_key.contiguous().reshape(bsz, self.num_heads, head_dim, h, w)
        proj_value = proj_value.contiguous().reshape(bsz, self.num_heads, head_dim, h, w)

        attn_weight = cca_weight(proj_query, proj_key, sampling_points)
        attn_weight = F.softmax(attn_weight, -1)

        output = cca_map(attn_weight, proj_value, sampling_points)
        output = self.output_conv(output)

        return output
