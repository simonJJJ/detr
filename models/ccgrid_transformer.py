import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules.cc_attention import RCCAModule, RCCAModuleNormal
from modules.grid_attention import GAModule
from modules.grid_align_attention import GaAlignModule


class CCGridTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = CCGridTransformerEncoderLayer(d_model, dim_feedforward,
                                                      dropout, activation, nhead)
        self.encoder = CCGridTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = CCGridTransformerDecoderLayer(d_model, dim_feedforward,
                                                      dropout, activation, nhead)
        self.decoder = CCGridTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        b, c, h, w = src.shape
        memory = self.encoder(src, pos_embed, mask)

        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(b, -1, -1)  # (b, num_queries, c)
        tgt = tgt.unsqueeze(0).expand(b, -1, -1)  # (b, num_queries, c)

        hs = self.decoder(tgt, memory, query_embed, mask)
        return hs


class CCGridTransformerEncoderLayer(nn.Module):
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
        src = src.transpose(1, 2).reshape(-1, c, h, w).contiguous()

        return src


class CCGridTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None, padding_mask=None):
        output = src  # (b, c, h, w)
        for layer in self.layers:
            output = layer(output, pos, padding_mask)

        return output


class CCGridTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()

        self.cross_attn = GaAlignModule(d_model)
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

    def forward(self, tgt, query_pos, src, src_padding_mask=None):
        bs = src.shape[0]
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)  # (b, num_queries, c)

        tgt_reshape = self.with_pos_embed(tgt, query_pos).transpose(1, 2).reshape(bs, self.d_model, 10, 10).contiguous()
        # (b * 4, c, 5, 5)
        #tgt_reshape = self.with_pos_embed(tgt, query_pos).transpose(1, 2).reshape(bs, self.d_model, 5, 5, 4).permute(0, 4, 1, 2, 3).flatten(0, 1).contiguous()
        tgt2, grid_weight = self.cross_attn(tgt_reshape, src, src_padding_mask)
        #tgt2 = tgt2.reshape(bs, 4, self.d_model, 5, 5).permute(0, 2, 3, 4, 1).flatten(2).transpose(1, 2)
        tgt2 = tgt2.flatten(2).transpose(1, 2)
        tgt = tgt + self.dropout1(tgt2)  # (b, num_queries, c)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, grid_weight


class CCGridTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_grid_weight = []
        for layer in self.layers:
            output, grid_weight = layer(output, query_pos, src, src_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_grid_weight.append(grid_weight)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_grid_weight)

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
    return CCGridTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=1024,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
