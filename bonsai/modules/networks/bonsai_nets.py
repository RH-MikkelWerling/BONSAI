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
        value_bin_vocab_size=0,
        value_embedding_mode="legacy",
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
            value_bin_vocab_size=value_bin_vocab_size,
            value_embedding_mode=value_embedding_mode,
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
            "value_bin_vocab_size": int(value_bin_vocab_size),
            "value_embedding_mode": value_embedding_mode,
        }

    def encode(self, batch, output_hidden_states: bool = False):
        """Encode a padded batch and return ``(batch, sequence, hidden)`` states.

        When ``output_hidden_states`` is True, also return the per-layer
        residual-stream outputs (pre-final-layernorm, one per transformer
        layer, unpacked to the padded ``(batch, sequence, hidden)`` layout)
        as a list, for inference-time pooling experiments over intermediate
        depths. This is additive: existing callers are unaffected because
        the default is False and the single-tensor return type is unchanged
        in that case.
        """
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
                value_bin=batch.get("value_bin"),
                value_normalized=batch.get("value_normalized"),
                value_present=batch.get("value_present"),
            )

        attention_mask = batch["attention_mask"].bool()
        use_flash = self.hparams["attn_type"] == "flash"
        hidden_size = x.shape[-1]

        # FlashAttention varlen operates on packed tokens. Packing once around
        # the transformer stack makes dynamically padded batches exact and
        # avoids spending attention compute on padding.
        if use_flash:
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
        hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
            )
            if output_hidden_states:
                layer_output = (
                    unpack_valid_tokens(x, attention_mask, hidden_size=hidden_size)
                    if use_flash
                    else x
                )
                hidden_states.append(layer_output)
        x = self.layernorm(x)

        if use_flash:
            x = unpack_valid_tokens(
                x,
                attention_mask,
                hidden_size=hidden_size,
            )

        if output_hidden_states:
            return x, hidden_states
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
        value_bin_vocab_size=0,
        value_embedding_mode="legacy",
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
            value_bin_vocab_size=value_bin_vocab_size,
            value_embedding_mode=value_embedding_mode,
        )
        self.pretrain_head = nn.Linear(hidden_size, vocab_size, bias=bias)
        self.value_embedding_mode = value_embedding_mode
        self.value_bin_head = None
        self.value_head = None
        if int(value_bin_vocab_size) > 0:
            self.value_bin_head = nn.Linear(
                hidden_size,
                int(value_bin_vocab_size),
                bias=bias,
            )
            self.value_head = nn.Linear(hidden_size, 1, bias=bias)

        # Weight tying (shares weights from code embedding to pretrain head)
        self.pretrain_head.weight = self.embeddings.code_embedding.weight

    def forward(self, batch: dict):
        last_hidden_state = super().forward(batch)
        labels = batch["target"]

        # Predicts only on the non-masked tokens
        mask = labels != -100
        code_hidden_state = last_hidden_state[mask]
        code_labels = labels[mask]

        logits = self.pretrain_head(code_hidden_state)
        if (
            self.value_bin_head is None
            or "target_value_mask" not in batch
            or "target_value_bin" not in batch
            or "target_value_normalized" not in batch
        ):
            return logits, code_labels

        value_mask = batch["target_value_mask"].bool()
        output = {
            "logits": logits,
            "labels": code_labels,
            "value_embedding_mode": self.value_embedding_mode,
        }
        if value_mask.any():
            value_hidden = last_hidden_state[value_mask]
            output["value_bin_logits"] = self.value_bin_head(value_hidden)
            output["target_value_bin"] = batch["target_value_bin"][value_mask].long()
            output["value_prediction"] = self.value_head(value_hidden).squeeze(-1)
            output["target_value_normalized"] = batch["target_value_normalized"][
                value_mask
            ].float()
        else:
            empty_hidden = last_hidden_state.reshape(-1, last_hidden_state.shape[-1])[
                :0
            ]
            output["value_bin_logits"] = self.value_bin_head(empty_hidden)
            output["target_value_bin"] = batch["target_value_bin"].reshape(-1)[:0]
            output["value_prediction"] = self.value_head(empty_hidden).squeeze(-1)
            output["target_value_normalized"] = batch[
                "target_value_normalized"
            ].reshape(-1)[:0]
        return output


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
        value_bin_vocab_size=0,
        value_embedding_mode="legacy",
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
            value_bin_vocab_size=value_bin_vocab_size,
            value_embedding_mode=value_embedding_mode,
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
