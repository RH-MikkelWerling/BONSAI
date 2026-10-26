import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "memorization_check",
    Path(__file__).parents[1] / "scripts" / "memorization_check.py",
)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


class LookupModel:
    """Reproduces training sequences whose prompt it has seen; else emits token 1."""

    def __init__(self, memorised):
        self.table = [list(s) for s in memorised]

    def generate(self, batch, max_new_tokens, start_after_n_tokens, **_):
        out = []
        for row in batch["event"].tolist():
            prompt = row[:start_after_n_tokens]
            match = next((s for s in self.table if s[: len(prompt)] == prompt), None)
            end = len(prompt) + max_new_tokens
            out.append(match[len(prompt) : end] if match else [1] * max_new_tokens)
        return {"generated": torch.tensor(out)}


def test_overlap_ignores_order_and_counts_each_code_once():
    truth, ages = np.array([1, 2, 3, 4]), np.array([0.0, 0.0, 1.0, 1.0])
    generated = np.array([3, 1, 1, 9])  # 3 a year early, 1 twice
    assert mc.count_reproduced(generated, truth, {1, 2, 3, 4}) == (2, 4)
    on_time = mc.count_reproduced(generated, truth, {1, 2, 3, 4}, ages, mc.ON_TIME_YEARS)
    assert on_time == (1, 4)


def test_generate_backend_separates_training_from_unseen_patients():
    rng = np.random.default_rng(0)
    records = {
        split: [{"subject_id": i, "code": rng.integers(2, 500, 80)} for i in range(20)]
        for split in ("train", "tuning")
    }
    model = LookupModel([r["code"] for r in records["train"]])

    def generate(patients, x, y):
        return mc.model_generate(
            model, patients, x, y, pad=0, device="cpu", batch_size=7
        )

    scores = mc.score_all_patients(
        generate, records, [8, 16], [1, 4, 16], 0, set(range(2, 500)), set()
    )
    excess = mc.excess_over_unseen(scores, "train", "tuning")
    for metric in ("exact_match", "overlap"):
        assert np.allclose(excess.filter(excess["metric"] == metric)["excess"], 1)


def _tiny_bonsai(vocab_size=24, max_seqlen=32):
    from bonsai.modules.networks.bonsai_nets import BonsaiPretrain

    return BonsaiPretrain(
        vocab_size=vocab_size,
        max_seqlen=max_seqlen,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=2,
        bias=False,
        dropout=0.0,
        attention_dropout=0.0,
        causal=True,
        attn_type="sdpa",
        abspos_encoding="fourier",
    )


def _subjects(n, length, seed, vocab_size=24):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        abspos = torch.cumsum(torch.rand(length, generator=g) * 50, 0).double()
        out.append(
            {
                "subject_id": seed * 1000 + i,
                "code": torch.randint(5, vocab_size, (length,), generator=g),
                "abspos": abspos,
                "age": 60 + abspos / 8766.0,
                "segment": torch.arange(length) // 3 + 1,
            }
        )
    return out


def test_bonsai_generation_is_step_by_step_and_detects_memorisation():
    torch.manual_seed(0)
    model = _tiny_bonsai()
    train, unseen = _subjects(6, 24, seed=1), _subjects(6, 24, seed=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = mc.bonsai_batch(train, 24, "cpu")
    for _ in range(300):  # memorise the six training sequences
        logits = model.pretrain_head(model.encode(batch)[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch["code"][:, 1:].reshape(-1)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.eval()

    # Against explicit step-by-step greedy decoding.
    prompt, steps = 8, 10
    generated = mc.bonsai_generate(model, train + unseen, prompt, steps, batch_size=4)
    for row, subject in enumerate(train + unseen):
        codes = subject["code"][:prompt].clone()
        with torch.inference_mode():
            for step in range(steps):
                part = mc.bonsai_batch(
                    [{**subject, "code": codes}], prompt + step, "cpu"
                )
                last = model.encode(part)[:, -1]
                codes = torch.cat([codes, model.pretrain_head(last).argmax(-1)])
        np.testing.assert_array_equal(generated[row], codes[prompt:].numpy())

    def generate(patients, x, y):
        return mc.bonsai_generate(model, patients, x, y, batch_size=4)

    scores = mc.score_all_patients(
        generate, {"train": train, "tuning": unseen}, [8], [8], 0, set(range(24)), set()
    )
    excess = mc.excess_over_unseen(scores, "train", "tuning")
    for metric in ("exact_match", "overlap", "on_time"):
        row = excess.filter(excess["metric"] == metric).row(0, named=True)
        assert row["train"] > 0.8 and row["unseen"] < 0.3 and row["excess_lower"] > 0
    losses = [mc.bonsai_losses(model, s, 24, batch_size=4) for s in (train, unseen)]
    check = mc.membership_from_loss(*losses).row(0, named=True)
    assert check["loss_gap_lower"] > 0 and check["membership_auc"] > 0.9


def test_summary_only_uses_cells_with_enough_scored_patients():
    import polars as pl

    rows = []
    for split, rare in (("train", 1.0), ("tuning", 0.0)):
        for i in range(60):
            # Every patient has an overlap score; only three have a rare code.
            rows.append({"split": split, "subject_id": str(i), "prompt_events": 16,
                         "continuation_events": 4, "exact_match": 0.0,
                         "overlap": 0.5 + 0.001 * i, "on_time": float("nan"),
                         "rare_codes": rare if i < 3 else float("nan")})
    excess = mc.excess_over_unseen(pl.DataFrame(rows), "train", "tuning")
    rare = excess.filter(pl.col("metric") == "rare_codes").row(0, named=True)
    assert rare["patients_train"] == 3 and rare["excess"] == 1.0
    summary = mc.write_summary(excess, None, min_patients=50)
    assert "rare_codes   no cell has enough patients" in summary
    assert "overlap      +0.000" in summary
