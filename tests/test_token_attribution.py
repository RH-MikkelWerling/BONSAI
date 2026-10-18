import pytest
import torch
from torch import nn

from bonsai.modules.networks.bonsai_nets import BonsaiBase
from opera.interpretability.token_attribution import (
    attributable_mask,
    baseline_tensor,
    borrowed_weight_target,
    event_ids,
    gradients_through_encoder,
    integrated_gradients,
    mean_content,
    replaced_target,
    risk_target,
    split_embeddings,
)
from opera.modules.networks.competing_risk import PiecewiseExponentialCompetingRiskLoss
from opera.modules.networks.source_anchored_readout import SourceAnchoredReadout

VOCAB = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "[MASK]": 4,
    "BG_SEX//F": 5,
    **{f"CODE{i}": 6 + i for i in range(10)},
}
FIXED = [0, 1, 2, 4]
OUTCOMES = ["overall_survival", "infection"]


class TinyModel(nn.Module):
    """The interface of the OPERA direct competing-risk model (cls_last pooling)."""

    def __init__(self, *, router=True, freeze=False):
        super().__init__()
        torch.manual_seed(3)
        self.encoder = BonsaiBase(
            vocab_size=len(VOCAB),
            max_seqlen=64,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=2,
            bias=False,
            dropout=0.1,
            attention_dropout=0.0,
            causal=True,
            attn_type="sdpa",
            abspos_encoding="scaled_time2vec",
        )
        self.competing_risk_loss = PiecewiseExponentialCompetingRiskLoss(
            OUTCOMES, [180, 365, 730], no_competing_outcomes=["overall_survival"]
        )
        self.competing_risk_head = nn.Linear(16, len(OUTCOMES) * 2 * 4)
        self.source_adapter_readout = (
            SourceAnchoredReadout(16, ["A", "B"], OUTCOMES, query_mode="interacting")
            if router
            else None
        )
        if router:
            with torch.no_grad():
                self.source_adapter_readout.up.normal_(std=0.3)
                self.source_adapter_readout.source_keys.normal_(std=1.0)
        self.freeze_encoder = freeze

    def get_objective_representations(self, batch):
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.freeze_encoder
        ):
            hidden = self.encoder(batch)
        lengths = batch["attention_mask"].sum(1) - 1
        return hidden[torch.arange(len(hidden)), lengths]

    def competing_risk_log_hazards(
        self, pooled, fine_index=None, *, removed_sources=()
    ):
        if self.source_adapter_readout is None:
            flat = self.competing_risk_head(pooled)
        else:
            adapted = self.source_adapter_readout(
                pooled, fine_index, removed_sources=removed_sources
            )
            weight = self.competing_risk_head.weight.reshape(len(OUTCOMES), 8, -1)
            bias = self.competing_risk_head.bias.reshape(len(OUTCOMES), 8)
            flat = torch.einsum("boh,okh->bok", adapted, weight) + bias[None]
        return flat.reshape(len(pooled), len(OUTCOMES), 2, 4)


def patient(codes, hours):
    n = len(codes)
    return {
        "code": torch.tensor(codes),
        "abspos": torch.tensor(hours, dtype=torch.float),
        "age": torch.linspace(60, 61, n),
        "segment": torch.arange(n) + 1,
        "attention_mask": torch.ones(n, dtype=torch.bool),
    }


def collate(samples):
    width = max(len(s["code"]) for s in samples)
    out = {}
    for key in samples[0]:
        out[key] = torch.stack(
            [torch.nn.functional.pad(s[key], (0, width - len(s[key]))) for s in samples]
        )
    out["cohort_fine_index"] = torch.tensor([0, 1, 0][: len(samples)])
    return out


SAMPLES = [
    patient([5, 6, 2, 7, 8, 2, 9, 1], [0, 10, 10, 50, 50, 50, 90, 100]),
    patient([5, 10, 11, 2, 12, 1], [0, 20, 30, 30, 70, 80]),
]


def setup(model, samples=SAMPLES):
    model.eval()
    batch = collate(samples)
    mask = attributable_mask(batch, FIXED)
    full, content = split_embeddings(model.encoder, batch)
    baseline = baseline_tensor(
        content, mask, mean_content(model.encoder, [batch], FIXED)
    )
    return batch, mask, full, content, baseline


def test_token_embeddings_reproduce_the_model_input():
    model = TinyModel()
    batch, mask, full, content, baseline = setup(model)
    target = risk_target(model, OUTCOMES, 365)
    with torch.no_grad():
        direct = target(batch)
        hooked = target({**batch, "token_embeddings": full})
    assert torch.allclose(direct, hooked, atol=1e-6)


@pytest.mark.parametrize("router", [True, False])
def test_attributions_add_up_to_the_risk_above_the_baseline(router):
    model = TinyModel(router=router)
    batch, mask, full, content, baseline = setup(model)
    attributions, f_input, f_base = integrated_gradients(
        risk_target(model, OUTCOMES, 365),
        batch,
        full,
        content,
        baseline,
        mask,
        steps=64,
    )
    gap = attributions.sum(-1) - (f_input - f_base)
    assert gap.abs().max() < 0.02 * (f_input - f_base).abs().max() + 1e-5
    # Separators, the prediction token and padding are never credited.
    assert torch.all(attributions.permute(0, 2, 1)[~mask] == 0)


def test_each_patient_is_explained_independently_of_the_batch():
    model = TinyModel()
    batch, mask, full, content, baseline = setup(model)
    together, _, _ = integrated_gradients(
        risk_target(model, OUTCOMES, 365),
        batch,
        full,
        content,
        baseline,
        mask,
        steps=8,
        chunk=3,
    )
    batch1, mask1, full1, content1, _ = setup(model, SAMPLES[1:])
    batch1["cohort_fine_index"] = batch["cohort_fine_index"][1:]
    base1 = baseline_tensor(
        content1, mask1, mean_content(model.encoder, [batch], FIXED)
    )
    alone, _, _ = integrated_gradients(
        risk_target(model, OUTCOMES, 365),
        batch1,
        full1,
        content1,
        base1,
        mask1,
        steps=8,
    )
    width = alone.shape[-1]
    assert torch.allclose(together[1, :, :width], alone[0], atol=1e-6)


def test_frozen_encoder_needs_the_gradient_switch():
    model = TinyModel(freeze=True)
    batch, mask, full, content, baseline = setup(model)
    with pytest.raises(RuntimeError):
        integrated_gradients(
            risk_target(model, OUTCOMES, 365),
            batch,
            full,
            content,
            baseline,
            mask,
            steps=4,
        )
    with gradients_through_encoder(model):
        attributions, _, _ = integrated_gradients(
            risk_target(model, OUTCOMES, 365),
            batch,
            full,
            content,
            baseline,
            mask,
            steps=4,
        )
    assert model.freeze_encoder and attributions.abs().sum() > 0


def test_replacing_every_event_gives_the_baseline_risk():
    model = TinyModel()
    batch, mask, full, content, baseline = setup(model)
    target = risk_target(model, OUTCOMES, 365)
    _, f_input, f_base = integrated_gradients(
        target, batch, full, content, baseline, mask, steps=2
    )
    replace = torch.stack([torch.zeros_like(mask), mask])
    values = replaced_target(target, batch, full, content, baseline, replace)
    assert torch.allclose(values[0], f_input, atol=1e-6)
    assert torch.allclose(values[1], f_base, atol=1e-6)


def test_events_group_tokens_recorded_at_the_same_time():
    batch = collate(SAMPLES)
    ids = event_ids(batch, attributable_mask(batch, FIXED))
    # Background, CODE0, [SEP], CODE1+CODE2 (same hour), [SEP], CODE3, [CLS].
    assert ids[0].tolist() == [0, 1, -1, 2, 2, -1, 3, -1]


def test_per_source_baseline_and_borrowed_weight_target():
    model = TinyModel()
    batch, mask, full, content, _ = setup(model)
    groups = lambda codes: (codes >= 10).long()  # noqa: E731
    _, by_group = mean_content(model.encoder, [batch], FIXED, groups=groups)
    baseline = baseline_tensor(
        content, mask, by_group, codes=batch["code"], groups=groups
    )
    assert torch.allclose(
        baseline[mask & (batch["code"] >= 10)].mean(0), by_group[1], atol=1e-6
    )
    borrowed = borrowed_weight_target(model, OUTCOMES)
    with torch.no_grad():
        values = borrowed({**batch, "token_embeddings": full})
    assert values.shape == (2, 2) and torch.all(
        (values >= -1e-6) & (values <= 1 + 1e-6)
    )


def _engine_sample(codes, hours, fine, sid):
    from torch import tensor

    sample = patient(codes, hours)
    sample.update(
        subject_id=sid,
        cohort_fine_index=tensor(fine),
        event_overall_survival=tensor(1),
        time_overall_survival=tensor(200.0),
        event_infection=tensor(-1),
        time_infection=tensor(-1.0),
    )
    return sample


COLUMNS = [
    ("risk", "overall_survival", 365.0),
    ("risk", "overall_survival", 730.0),
    ("risk", "infection", 365.0),
    ("borrowed", "overall_survival", None),
    ("logit", "overall_survival", 365.0),
    ("logit", "infection", 365.0),
]


def _explainer(model, **kwargs):
    from opera.modules.datamodules.ContrastiveDataModule import contrastive_collate
    from opera.run.explain_direct_competing_risk import Explainer

    explainer = Explainer(
        model,
        VOCAB,
        collate=contrastive_collate,
        columns=COLUMNS,
        check_columns=[COLUMNS[0]],
        steps=16,
        ks=(1, 2),
        random_repeats=3,
        token_budget=64,
        **kwargs,
    )
    explainer.fit_baseline([collate(SAMPLES)])
    return explainer


def test_one_pass_explains_every_column_and_matches_single_column_runs():
    from opera.run.explain_direct_competing_risk import column_name, randomized_copy

    model = TinyModel()
    explainer = _explainer(model)
    sample = _engine_sample(
        SAMPLES[0]["code"].tolist(), SAMPLES[0]["abspos"].tolist(), 0, 10
    )
    sanity = [
        ("random_head", randomized_copy(model, "head")),
        ("random_model", randomized_copy(model, "model")),
    ]
    rows = explainer.explain(
        sample,
        meta={"subject_id": 10, "weight": 2.0},
        sanity_models=sanity,
        alternative_baseline="source_mean",
    )
    # Not eligible for infection: that column is NaN, the others are filled.
    (tokens,) = rows["tokens"]
    assert tokens[column_name(*COLUMNS[2])].isna().all()
    assert tokens[column_name(*COLUMNS[0])].notna().all() and len(tokens) == 5
    assert {r["column"] for r in rows["patients"]} == {
        column_name(*c) for c in (COLUMNS[0], COLUMNS[1], COLUMNS[3], COLUMNS[4])
    }
    for summary in rows["patients"]:
        assert abs(summary["completeness_gap"]) <= 0.05
    assert sorted(round(d * 24) for d in tokens.days_before_index) == [
        10,
        50,
        50,
        90,
        100,
    ]
    # Checks only for the check column: k=1,2 x (top, bottom, 3 random) x (replace, delete).
    assert len(rows["faithfulness"]) == 2 * 5 * 2
    assert {r["check"] for r in rows["sanity"]} == {
        "random_head",
        "random_model",
        "baseline_source_mean",
    }
    # The shared pass gives the same attributions as explaining one column alone.
    single = _explainer(model)
    single.columns = [COLUMNS[1]]
    (alone,) = single.explain(sample, meta={"subject_id": 10})["tokens"]
    name = column_name(*COLUMNS[1])
    assert torch.allclose(
        torch.tensor(alone[name].to_numpy()),
        torch.tensor(tokens[name].to_numpy()),
        atol=1e-6,
    )


def test_stratified_sample_weights_restore_the_population():
    from opera.run.explain_direct_competing_risk import stratified_sample

    pool = [{"subtype": "rare"}] * 10 + [{"subtype": "common"}] * 990
    chosen, weights = stratified_sample(pool, per_subtype=10, total=110, seed=1)
    assert len(chosen) == 110 and len(set(chosen)) == 110
    assert sum(1 for i in chosen if i < 10) == 10  # every rare patient is in
    assert abs(sum(weights) - 1000) < 1e-6  # weights sum to the population
    assert weights[chosen.index(0)] == 1.0


def test_tables_on_disk_summarise_into_every_exported_table(tmp_path):
    import json

    import pandas as pd

    from opera.run.explain_direct_competing_risk import column_name
    from opera.run.summarise_attributions import run

    model = TinyModel()
    explainer = _explainer(model)
    tables = {"tokens": [], "patients": [], "faithfulness": [], "sanity": []}
    for i in range(30):
        base = SAMPLES[i % 2]
        sample = _engine_sample(
            base["code"].tolist(), base["abspos"].tolist(), i % 2, 100 + i
        )
        sample["code"][1] = 6 + (i % 10)  # vary one event between patients
        rows = explainer.explain(
            sample,
            meta={
                "subject_id": 100 + i,
                "cohort": "X",
                "subtype": "AB"[i % 2],
                "weight": 1.0 + i % 2,
            },
        )
        for key in tables:
            tables[key] += rows[key]
    (tmp_path / "token_attributions_INTERNAL").mkdir()
    pd.concat(tables["tokens"]).to_parquet(
        tmp_path / "token_attributions_INTERNAL/part-00000.parquet"
    )
    for key, name in (
        ("patients", "patient_attributions"),
        ("faithfulness", "faithfulness"),
        ("sanity", "sanity"),
    ):
        pd.DataFrame(tables[key]).to_parquet(tmp_path / f"{name}_INTERNAL.parquet")
    (tmp_path / "lab_bin_counts.json").write_text("{}")
    (tmp_path / "attribution_run.json").write_text(
        json.dumps(
            {
                "columns": [column_name(*c) for c in COLUMNS],
                "check_columns": [column_name(*COLUMNS[0])],
                "split": "held_out",
                "n_patients": 30,
                "steps": 16,
                "baseline": "mean",
                "tolerance": 0.05,
            }
        )
    )
    config = {
        "summary": {"bootstrap_draws": 50, "min_count": 5, "horizon_days": 365},
        "concepts": [{"name": "code0", "label": "Code 0", "codes": ["CODE0"],
                      "expected": {"overall_survival": "+"}},
                     {"name": "code1", "label": "Code 1", "codes": ["CODE1"]}],
    }
    out = tmp_path / "out"
    numbers = run(tmp_path, config, out)
    # Findings x outcomes on the log-odds (atlas) columns; only expected cells are tests.
    concepts = pd.read_csv(out / "attribution_concepts.csv")
    assert set(concepts.scale) == {"logit"}
    tested = concepts.loc[concepts.expected.notna()]
    assert list(tested.concept) == ["code0"] and list(tested.outcome) == ["overall_survival"]
    assert set(concepts.loc[concepts.expected.isna(), "verdict"]) <= {
        "descriptive", "too few patients"}
    assert len(pd.read_csv(out / "attribution_faithfulness.csv"))
    assert len(pd.read_csv(out / "attribution_top_concepts.csv"))
    assert numbers["ig_patients"] == 30 and numbers["ig_outcomes"] == 1
    variance = pd.read_csv(out / "attribution_variance_sources.csv")
    assert set(variance.outcome) == {"overall_survival"} and set(variance.scale) == {"logit"}
    assert abs(variance.share.sum() - 1) < 1e-6  # exact decomposition
    assert (out / "attribution_outcome_similarity.csv").exists()
    assert "ig_var_reference" in numbers
    with pytest.raises(ValueError, match="Select one"):
        run(tmp_path, {**config, "summary": {"bootstrap_draws": 5, "min_count": 5}},
            tmp_path / "ambiguous")
    unreviewed = {**config, "concept_mapping_reviewed": False,
                  "summary": {**config["summary"], "horizon_days": 730}}
    with pytest.warns(UserWarning, match="unreviewed"):
        run(tmp_path, unreviewed, tmp_path / "730")
    assert pd.read_csv(tmp_path / "730/attribution_concepts.csv").empty
    from opera.visualization.paper_figures import (
        figure_attribution_atlas, figure_attribution_structure)
    import matplotlib.pyplot as plt

    fig = figure_attribution_atlas(variance, pd.read_csv(out / "attribution_variance_windows.csv"),
                                   concepts)
    plt.close(fig)
    fig = figure_attribution_structure(pd.read_csv(out / "attribution_outcome_similarity.csv"))
    plt.close(fig)


def test_patient_perturbations_are_independent_of_prior_explanation_order():
    model = TinyModel()
    explainer = _explainer(model)
    first = _engine_sample(
        SAMPLES[0]["code"].tolist(), SAMPLES[0]["abspos"].tolist(), 0, 8
    )
    other = _engine_sample(
        SAMPLES[1]["code"].tolist(), SAMPLES[1]["abspos"].tolist(), 1, 9
    )
    a = explainer.explain(first, meta={"subject_id": 8, "weight": 1.0})
    explainer.explain(other, meta={"subject_id": 9, "weight": 1.0})
    b = explainer.explain(first, meta={"subject_id": 8, "weight": 1.0})
    assert a["faithfulness"] == b["faithfulness"]


def test_real_opera_checkpoint_loads_flash_metadata_under_sdpa_without_layout_change(
    tmp_path,
):
    from omegaconf import OmegaConf
    from opera.modules.networks.opera_nets import OperaContrastiveModel
    from opera.diagnostics.representation_gradient_conflict import _load_model

    encoder = TinyModel(router=False).encoder
    grids = {o: torch.tensor([30.0, 180.0, 365.0]) for o in OUTCOMES}
    probabilities = {o: torch.tensor([0.3, 0.3, 0.4]) for o in OUTCOMES}
    cr = {
        "loss_weight": 1.0,
        "contrastive_loss_weight": 0.0,
        "interval_boundaries_days": [180.0, 365.0, 730.0],
        "no_competing_outcomes": ["overall_survival"],
    }
    model = OperaContrastiveModel(
        encoder,
        OUTCOMES,
        hidden_size=16,
        projection_hidden_dim=16,
        projection_dim=8,
        outcome_sorted_event_times=grids,
        outcome_event_time_probs=probabilities,
        competing_risk_config=cr,
        dapt_anchor_weight=0.0,
        pooling="cls_last",
    ).eval()
    saved_config = {**encoder.hparams, "attn_type": "flash"}
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {
                "model_config": saved_config,
                "model_init_config": model.model_init_config,
            },
        },
        path,
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "projection_hidden_dim": 16,
                "projection_dim": 8,
                "temperature": 0.07,
            },
            "cohorts": {"A": {}},
            "competing_risk": cr,
            "cross_outcome": {},
        }
    )
    loaded = _load_model(
        path,
        cfg,
        OUTCOMES,
        grids,
        probabilities,
        torch.device("cpu"),
        execution_attn_type="sdpa",
    )
    assert loaded.encoder.hparams["attn_type"] == "sdpa"
    assert (
        loaded.encoder.hparams["rope_interleaved"] == saved_config["rope_interleaved"]
    )
    batch = collate(SAMPLES)
    with torch.no_grad():
        torch.testing.assert_close(
            risk_target(model, OUTCOMES, 365)(batch),
            risk_target(loaded, OUTCOMES, 365)(batch),
        )
    _, mask, full, content, baseline = setup(loaded)
    attrs, f, base = integrated_gradients(
        risk_target(loaded, OUTCOMES, 365),
        batch,
        full,
        content,
        baseline,
        mask,
        steps=64,
    )
    assert torch.isfinite(attrs).all()
    assert (attrs.sum(-1) - (f - base)).abs().max() < 1e-3


def test_log_odds_attributions_are_complete_and_share_the_sign_of_risk_attributions():
    model = TinyModel()
    batch, mask, full, content, baseline = setup(model)
    from opera.interpretability.token_attribution import combined_target

    columns = [("risk", "infection", 365.0), ("logit", "infection", 365.0)]
    attributions, f_input, f_base = integrated_gradients(
        combined_target(model, columns), batch, full, content, baseline, mask, steps=64)
    gap = attributions.sum(-1) - (f_input - f_base)
    assert gap.abs().max() < 0.02 * (f_input - f_base).abs().max() + 1e-5
    assert torch.allclose(torch.logit(f_input[:, 0]), f_input[:, 1], atol=1e-5)
    risk, logit = attributions[:, 0][mask], attributions[:, 1][mask]
    large = risk.abs() > risk.abs().median()
    assert (torch.sign(risk[large]) == torch.sign(logit[large])).float().mean() > 0.9


def test_clinical_explanations_render_from_a_cached_run(tmp_path, monkeypatch):
    import json
    import sys

    import pandas as pd

    from opera.run import render_individual_attribution
    from opera.run.explain_direct_competing_risk import column_name

    explainer = _explainer(TinyModel())
    tables = {"tokens": [], "patients": []}
    for i in range(6):
        base = SAMPLES[i % 2]
        sample = _engine_sample(base["code"].tolist(), base["abspos"].tolist(), i % 2, 200 + i)
        rows = explainer.explain(sample, meta={"subject_id": 200 + i, "weight": 1.0})
        tables["tokens"] += rows["tokens"]
        tables["patients"] += rows["patients"]
    (tmp_path / "token_attributions_INTERNAL").mkdir()
    pd.concat(tables["tokens"]).to_parquet(
        tmp_path / "token_attributions_INTERNAL/part-00000.parquet")
    pd.DataFrame(tables["patients"]).to_parquet(tmp_path / "patient_attributions_INTERNAL.parquet")
    (tmp_path / "lab_bin_counts.json").write_text("{}")
    (tmp_path / "attribution_run.json").write_text(json.dumps(
        {"columns": [column_name(*c) for c in COLUMNS], "tolerance": 0.05}))
    out = tmp_path / "local"
    monkeypatch.setattr(sys, "argv", [
        "render", "--input-dir", str(tmp_path), "--subject-id", "201", "--all",
        "--output-dir", str(out)])
    render_individual_attribution.main()
    assert (out / "patient_201_INTERNAL.png").exists()
    cards = pd.read_csv(out / "patient_201_INTERNAL.csv")
    assert set(cards.outcome) == {"overall_survival"}
    table = pd.read_parquet(out / "explanations_INTERNAL.parquet")
    assert set(table.subject_id) == {200 + i for i in range(6)}
    assert set(table.direction) <= {"raises", "lowers"}
