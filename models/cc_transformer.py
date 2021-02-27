import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules.cc_attention import RCCAModule
from modules.cross_cc_attention import CrossCCAttention


class CCTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = CCTransformerEncoderLayer(d_model, dim_feedforward,
                                                  dropout, activation, nhead)
        self.encoder = CCTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = CCTransformerDecoderLayer(d_model, dim_feedforward,
                                                  dropout, activation, nhead)
        self.decoder = CCTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.sampling_points = nn.Linear(d_model, 2)

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.sampling_points.weight.data, gain=1.0)
        nn.init.constant_(self.sampling_points.bias.data, 0.)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio  # (b, 2)

    def forward(self, src, mask, query_embed, pos_embed):
        b, c, h, w = src.shape
        memory = self.encoder(src, pos_embed, mask)

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in [mask]], 1)  # (b, 1, 2)

        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(b, -1, -1)  # (b, num_queries, c)
        tgt = tgt.unsqueeze(0).expand(b, -1, -1)  # (b, num_queries, c)
        sampling_points = self.sampling_points(query_embed).sigmoid()  # (b, num_queries, 2)
        num_query = sampling_points.shape[1]
        sampling_points_input = sampling_points.unsqueeze(2).expand(b, num_query, self.nhead, 2)

        hs = self.decoder(tgt, sampling_points_input, memory, valid_ratios, query_embed, pos_embed, mask)
        return hs, sampling_points


class CCTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn=1024,
                 dropout=0.1, activation="relu", n_heads=8):
        super().__init__()

        self.self_attn = RCCAModule(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, padding_mask=None):
        b, c, h, w = src.shape
        src2 = self.self_attn(self.with_pos_embed(src, pos), src, pos, padding_mask)  # (b, c, h, w)
        src2 = src2.flatten(2).transpose(1, 2)  # (b, h*w, c)
        src = src.flatten(2).transpose(1, 2)  # (b, h*w, c)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)
        src = src.transpose(1, 2).reshape(b, c, h, w).contiguous()

        return src


class CCTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None, padding_mask=None):
        output = src  # (b, c, h, w)
        for layer in self.layers:
            output = layer(output, pos, padding_mask)

        return output


class CCTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()

        self.cross_attn = CrossCCAttention(d_model, num_heads=n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.d_model = d_model

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, sampling_points, query_pos, src, pos_embed=None, src_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)  # (b, num_queries, c)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), self.with_pos_embed(src, pos_embed), src, sampling_points, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)  # (b, num_queries, c)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class CCTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, sampling_points, src, src_valid_ratios, query_pos=None, pos_embed=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        for layer in self.layers:
            #reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            sampling_points_input = sampling_points * src_valid_ratios[:, None]
            output = layer(output, sampling_points_input, query_pos, src, pos_embed, src_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.squeeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    return CCTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=1024,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
