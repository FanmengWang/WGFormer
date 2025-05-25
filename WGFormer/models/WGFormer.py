# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from typing import Optional
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture

logger = logging.getLogger(__name__)


@register_model("WGFormer")
class WGFormerModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--distance-loss",
            type=float,
            default=8.0,
            help="weight for the distance loss",
        )
        parser.add_argument(
            "--coord-loss",
            type=float,
            default=1.0,
            help="weight for the coordinate loss",
        )
        parser.add_argument(
            "--num-sinkhorn-iteration",
            type=int,
            default=3,
            help="number of iteration to use sinkhorn",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        WGFormer_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.backbone = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        
    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        **kwargs
    ):
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        def single_encoder(
            emb: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
        ):
            x = self.backbone.emb_layer_norm(emb)
            x = F.dropout(x, p=self.backbone.emb_dropout, training=self.training)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            attn_mask, padding_mask = fill_attn_mask(
                attn_mask, padding_mask, fill_val=float("-inf")
            )
            for i in range(len(self.backbone.layers)):
                x, attn_mask, _ = self.backbone.layers[i](
                    x, self.args.num_sinkhorn_iteration, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )
            return x, attn_mask

        padding_mask = src_tokens.eq(self.padding_idx)
        input_padding_mask = padding_mask
        
        attn_mask = get_dist_features(src_distance, src_edge_type)
        input_attn_mask = attn_mask
        
        x = self.embed_tokens(src_tokens)
        bsz = x.size(0)
        seq_len = x.size(1)
        
        x, attn_mask = single_encoder(
            x, padding_mask=padding_mask, attn_mask=attn_mask
        )

        if self.backbone.final_layer_norm is not None:
            x = self.backbone.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask, _ = fill_attn_mask(attn_mask, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        if padding_mask is not None:
            atom_num = (torch.sum(~padding_mask, dim=1) - 1).view(-1, 1, 1, 1)
        else:
            atom_num = src_coord.shape[1] - 1
        delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
        attn_probs = self.pair2coord_proj(delta_pair_repr)
        coords_update = delta_pos / atom_num * attn_probs
        coords_update = torch.sum(coords_update, dim=2)
        coords_predict = src_coord + coords_update

        distance_predict = torch.sqrt(torch.clamp(torch.sum((coords_predict.unsqueeze(2) - coords_predict.unsqueeze(1))**2, dim=-1), min=1e-12))
        
        return [distance_predict, coords_predict]


class NonLinearHead(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("WGFormer", "WGFormer")
def WGFormer_architecture(args):
    def base_architecture(args):
        args.encoder_layers = getattr(args, "encoder_layers", 30)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
        args.max_seq_len = getattr(args, "max_seq_len", 512)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.coord_loss = getattr(args, "coord_loss", 1.0)
        args.distance_loss = getattr(args, "distance_loss", 1.0)
        args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    base_architecture(args)
