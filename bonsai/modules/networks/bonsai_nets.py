import torch
import torch.nn as nn

from bonsai.functional.model_config import NATIVE_ARCHITECTURE_VERSION
from bonsai.modules.networks.components.embeddings import EhrEmbeddings
from bonsai.modules.networks.components.layer import TransformerLayer, attn_types

_FLASH_ATTENTION_AVAILABLE = "flash" in attn_types


def pack_valid_tokens(x: torch.Tensor, attention_mask: torch.Tensor):
    """Pack padded token states and return FlashAttention sequence offsets."""
    attention_mask = attention_mask.bool()
    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    if torch.any(lengths <= 0):
        raise ValueError("Every sequence must contain at least one valid token.")
    cu_seqlens = torch.zeros(
        x.shape[0] + 1,
        dtype=torch.int32,
        device=x.device,
    )
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return x[attention_mask], cu_seqlens


def unpack_valid_tokens(
    packed: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """Restore packed token states to the original padded batch shape."""
    output = packed.new_zeros(
        attention_mask.shape[0],
        attention_mask.shape[1],
        hidden_size,
    )
    output[attention_mask.bool()] = packed
    return output


class BonsaiBase(nn.Module):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
    ):
        if attn_type == "flash" and not _FLASH_ATTENTION_AVAILABLE:
            raise ImportError(
                "flash_attn is not available. Please install flash-attn or use `attn_type='sdpa'` instead."
            )
        super().__init__()
        self.embeddings = EhrEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seqlen=max_seqlen,
        )
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    bias=bias,
                    max_seqlen=max_seqlen,
                    causal=causal,
                    attn_type=attn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(hidden_size, bias=bias)

        self.hparams = {
            "architecture_version": NATIVE_ARCHITECTURE_VERSION,
            "vocab_size": vocab_size,
            "max_seqlen": max_seqlen,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "bias": bias,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "causal": causal,
            "attn_type": attn_type,
        }

    def encode(self, batch):
        """Encode a padded batch and return ``(batch, sequence, hidden)`` states."""
        if "token_embeddings" in batch:
            x = batch["token_embeddings"]
            if x.shape[:2] != batch["code"].shape:
                raise ValueError(
                    "token_embeddings must match batch['code'] in batch and "
                    "sequence dimensions."
                )
            if x.shape[-1] != self.hparams["hidden_size"]:
                raise ValueError(
                    "token_embeddings hidden dimension does not match the encoder."
                )
        else:
            x = self.embeddings(
                code=batch["code"],
                age=batch["age"],
                abspos=batch["abspos"],
                segment=batch["segment"],
            )

        attention_mask = batch["attention_mask"].bool()
        use_flash = self.hparams["attn_type"] == "flash"

        # FlashAttention varlen operates on packed tokens. Packing once around
        # the transformer stack makes dynamically padded batches exact and
        # avoids spending attention compute on padding.
        if use_flash:
            hidden_size = x.shape[-1]
            x, cu_seqlens = pack_valid_tokens(x, attention_mask)
            attn_mask = None
        # SDPA requires a broadcasted boolean attention mask.
        elif not self.hparams["causal"]:
            attn_mask = attention_mask[:, None, None, :]
            cu_seqlens = None
        else:
            attn_mask = None
            cu_seqlens = None

        x = self.drop(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
            )
        x = self.layernorm(x)

        if use_flash:
            x = unpack_valid_tokens(
                x,
                attention_mask,
                hidden_size=hidden_size,
            )

        return x

    def forward(self, batch):
        return self.encode(batch)


class BonsaiPretrain(BonsaiBase):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            bias=bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            attn_type=attn_type,
        )
        self.pretrain_head = nn.Linear(hidden_size, vocab_size, bias=bias)

        # Weight tying (shares weights from code embedding to pretrain head)
        self.pretrain_head.weight = self.embeddings.code_embedding.weight

    def forward(self, batch: dict):
        last_hidden_state = super().forward(batch)
        labels = batch["target"]

        # Predicts only on the non-masked tokens
        mask = labels != -100
        last_hidden_state = last_hidden_state[mask]
        labels = labels[mask]

        logits = self.pretrain_head(last_hidden_state)
        return logits, labels


class BonsaiFinetune(BonsaiBase):
    def __init__(
        self,
        # Embedding / vocab
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_layers,
        num_attention_heads,
        # Attention / behavior
        bias,
        dropout,
        attention_dropout,
        causal,
        attn_type,
        # Misc
        predict_token_id,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            bias=bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            causal=causal,
            attn_type=attn_type,
        )
        self.hparams["predict_token_id"] = predict_token_id
        self.finetune_head = nn.Linear(hidden_size, 1, bias=bias)

    def get_pooled_representation(self, batch: dict):
        """Return the native prediction-token representation for each subject."""
        last_hidden_state = self.encode(batch)
        pred_tokens = batch["code"] == self.hparams["predict_token_id"]
        counts = pred_tokens.sum(dim=1)
        if not torch.all(counts == 1):
            raise ValueError(
                "Every finetuning sequence must contain exactly one prediction token."
            )
        return last_hidden_state[pred_tokens]

    def forward(self, batch: dict):
        return self.finetune_head(self.get_pooled_representation(batch))
