import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location(
    "memorization_check", Path(__file__).parents[1] / "scripts" / "memorization_check.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


class LookupModel:
    """Reproduces training sequences whose prompt it has seen; else emits token 1."""

    def __init__(self, memorised):
        self.table = {tuple(s): s for s in memorised}

    def generate(self, batch, max_new_tokens, temperature, pad_token_id,
                 start_after_n_tokens, n_samples_per_person):
        events = batch["event"]
        assert (events[:, start_after_n_tokens + max_new_tokens:] == pad_token_id).all()
        out = []
        for row in events.tolist():
            prompt = tuple(row[:start_after_n_tokens])
            match = next((s for k, s in self.table.items() if k[:len(prompt)] == prompt), None)
            cont = match[len(prompt):len(prompt) + max_new_tokens] if match else [1] * max_new_tokens
            out.append(cont)
        return {"generated": torch.tensor(out)}


def make_frame(n=40, length=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"subject_id": i, "split": "train" if i < n // 2 else "tuning",
             "event": rng.integers(2, 500, size=length).tolist()} for i in range(n)]
    return pd.DataFrame(rows)


def test_training_patients_match_unseen_do_not():
    frame = make_frame()
    model = LookupModel(frame.loc[frame.split == "train", "event"])
    scores = mc.run(model, frame, splits=["train", "tuning"], prompts=[8, 16], news=[1, 4, 16],
                    n=0, pad=0, device="cpu", batch_size=7)
    summary = mc.summarise(scores)
    train = summary.loc[summary.split == "train"]
    held = summary.loc[summary.split == "tuning"]
    assert (train.exact_match == 1).all() and (held.exact_match == 0).all()
    assert len(summary) == 2 * 2 * 3
    ex = mc.excess(summary, "train", "tuning")
    assert np.allclose(ex.excess, 1)


def test_score_prefix_and_ignore():
    s = mc.score([[5, 6, 7], [5, 9, 7]], [[5, 6, 7], [5, 6, 7]], ignore=frozenset({5}))
    assert s.exact.tolist() == [True, False]
    assert s.correct_prefix.tolist() == [3, 1]
    assert np.isclose(s.content_agreement[1], 0.5)


def test_short_patients_are_excluded():
    frame = make_frame(length=10)
    scores = mc.run(LookupModel([]), frame, splits=["train"], prompts=[8], news=[4],
                    n=0, pad=0, device="cpu", batch_size=4)
    assert scores.empty


def _tiny_bonsai(vocab_size=24, max_seqlen=32):
    from bonsai.modules.networks.bonsai_nets import BonsaiPretrain
    return BonsaiPretrain(vocab_size=vocab_size, max_seqlen=max_seqlen, hidden_size=32,
                          num_layers=2, num_attention_heads=2, bias=False, dropout=0.0,
                          attention_dropout=0.0, causal=True, attn_type="sdpa",
                          abspos_encoding="fourier")


def _subjects(n, length, seed, vocab_size=24):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        abspos = torch.cumsum(torch.rand(length, generator=g) * 50, 0).double()
        out.append({"subject_id": f"{seed}-{i}",
                    "code": torch.randint(5, vocab_size, (length,), generator=g),
                    "abspos": abspos, "age": 60 + abspos / 8766.0,
                    "segment": torch.arange(length) // 3 + 1})
    return out


def test_bonsai_single_pass_equals_greedy_generation_and_detects_memorisation():
    torch.manual_seed(0)
    model = _tiny_bonsai()
    train, unseen = _subjects(6, 24, seed=1), _subjects(6, 24, seed=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = mc.bonsai_batch(train, 24, "cpu")
    for _ in range(300):  # memorise the six training sequences
        logits = model.pretrain_head(model.encode(batch)[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 batch["code"][:, 1:].reshape(-1))
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    model.eval()

    # Explicit greedy generation with the patients' own event times.
    prompt, new = 8, 10
    hits = mc.next_code_hits(model, train + unseen, prompt + new, device="cpu", batch_size=4)
    for row, subject in enumerate(train + unseen):
        generated = subject["code"][:prompt].clone()
        for step in range(new):
            width = prompt + step
            part = {**subject, "code": generated}
            with torch.inference_mode():
                hidden = model.encode(mc.bonsai_batch([part], width, "cpu"))
                generated = torch.cat([generated, model.pretrain_head(hidden[:, -1]).argmax(-1)])
        matches = (generated[prompt:] == subject["code"][prompt:prompt + new]).numpy()
        scored = mc.score_hits(hits[[row], prompt - 1:prompt - 1 + new],
                               subject["code"][prompt:prompt + new].numpy()[None])
        first_miss = int(np.argmin(matches)) if not matches.all() else new
        assert bool(scored.exact[0]) == bool(matches.all())
        assert int(scored.correct_prefix[0]) == first_miss

    scores = mc.run_bonsai(model, {"train": train, "tuning": unseen}, prompts=[4, 8],
                           news=[1, 8], n=0, device="cpu", batch_size=4)
    summary = mc.summarise(scores).set_index(["split", "prompt_tokens", "new_tokens"])
    assert summary.loc[("train", 8, 8), "exact_match"] > 0.8
    assert summary.loc[("tuning", 8, 8), "exact_match"] < 0.2
