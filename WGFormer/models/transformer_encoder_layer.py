# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from unicore import utils
from typing import Dict, Optional
from unicore.modules import LayerNorm
from .multihead_attention import SelfMultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln

    def forward(
        self,
        x: torch.Tensor,
        n_it: int,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            n_it=n_it,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs
                