import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

from bonsai.modules.networks.components.rope_fa import FlashRotaryEmbedding


class FlashMultiHeadAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, attention_dropout, bias, max_seqlen, causal
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"Hidden size {hidden_size} must be divisible by num_heads {num_heads} "
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seqlen = max_seqlen

        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.rotary_embedding = FlashRotaryEmbedding(dim=self.head_dim)
        self.self_attn = FlashSelfAttention(
            causal=causal, attention_dropout=attention_dropout
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, cu_seqlens=None, **kwargs):
        qkv = self.Wqkv(x)

        if cu_seqlens is None:
            if x.dim() != 3:
                raise ValueError("Padded FlashAttention input must be rank 3.")
            batch_size, seqlen, _ = x.shape
            qkv = qkv.view(batch_size, seqlen, 3, self.num_heads, self.head_dim)
            max_seqlen = seqlen
        else:
            if x.dim() != 2:
                raise ValueError("Packed FlashAttention input must be rank 2.")
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
            max_seqlen = int(torch.diff(cu_seqlens).max().item())

        qkv = self.rotary_embedding(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        y = y.reshape(*x.shape[:-1], -1)
        y = self.out_proj(y)
        return y
