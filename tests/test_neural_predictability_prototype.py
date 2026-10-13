"""Regression checks for the validation-only neural predictability plot."""

import io
import json
import tarfile

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from opera.run import plot_neural_predictability_prototype as neural


def test_archive_merges_resumed_lightning_metrics() -> None:
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        for version, value in ((0, "0,0.7\n"), (1, "1,0.8\n")):
            data = ("epoch,val/cr_cif_auc_endpoint_group_macro\n" + value).encode()
            name = f"recent_validation/runs/root1/example/contrastive_multicohort_runs/version_{version}/metrics.csv"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    memory.seek(0)
    with tarfile.open(fileobj=memory, mode="r:gz") as archive:
        frame = neural._archive_csv(archive, "example", "metrics.csv")
    assert frame["epoch"].tolist() == [0, 1]


def test_approved_run_atlas_merges_resumed_versions(tmp_path) -> None:
    run = tmp_path / "router"
    for version, epoch, monitor, auc in ((0, 0, .6, .7), (1, 1, .8, .9)):
        folder = run / "contrastive_multicohort_runs" / f"version_{version}"
        folder.mkdir(parents=True)
        pd.DataFrame({"epoch": [epoch],
                      "val/cr_cif_auc_endpoint_group_macro": [monitor],
                      "val/cr_cif_auc/x/365d": [auc]}).to_csv(folder / "metrics.csv", index=False)
    (run / "training_complete.json").write_text('{"status":"completed"}')
    (run / "endpoint_selection_inventory.json").write_text(json.dumps([
        {"outcome": "x", "horizon_days": 365, "n_cases": 25,
         "n_controls": 40, "included": True},
    ]))
    pd.DataFrame({"outcome": ["x"], "end_day": [365],
                  "n_target": [50]}).to_csv(run / "competing_risk_interval_support.csv", index=False)
    registry = tmp_path / "registry.yaml"
    registry.write_text("outcomes:\n  x: {}\n")
    groups = tmp_path / "groups.yaml"
    groups.write_text("Clinical:\n  - x\n")
    frame, epoch = neural.prepare_approved_run_data(
        run, registry_path=registry, groups_path=groups, horizons=(365,)
    )
    assert epoch == 1
    assert frame.iloc[0]["cif_auc"] == pytest.approx(.9)
    assert frame.iloc[0]["training_primary_events"] == 50
    assert frame.iloc[0]["tuning_primary_events"] == 25


def test_plot_shows_low_auc_and_group_iqr(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "outcome": ["a", "b", "c", "a", "b", "c"],
            "display_group": ["Group"] * 6,
            "horizon_days": [30, 30, 30, 365, 365, 365],
            "cif_auc": [0.06, 0.7, 0.9, 0.6, 0.75, 0.85],
            "training_primary_events": [2, 3, 4, 7, 8, 9],
        }
    )
    captured = {}

    def capture(fig, path):
        captured["ymin"] = fig.axes[0].get_ylim()[0]
        captured["group_lines"] = len(fig.axes[1].lines)

    monkeypatch.setattr(neural, "save_fig", capture)
    neural.plot_neural_predictability(
        frame, neural.Path("unused.png"), run_label="test", selected_epoch=0
    )
    assert captured["ymin"] <= 0.06
    assert captured["group_lines"] == 2  # one IQR bar per horizon
    assert not plt.get_fignums()


def test_horizon_lines_fit_log_event_count_within_each_horizon(monkeypatch) -> None:
    rows = []
    for horizon in (30, 365):
        for index in range(24):
            rows.append({
                "outcome": f"outcome_{index}", "display_group": "Group",
                "horizon_days": horizon, "training_primary_events": index + 2,
                "cif_auc": 0.55 + index / 100 + horizon / 10000,
            })
    frame = pd.DataFrame(rows)
    captured = {}

    def capture(fig, path):
        captured["lines"] = list(fig.axes[0].lines)
        captured["group_lines"] = len(fig.axes[1].lines)

    monkeypatch.setattr(neural, "save_fig", capture)
    neural.plot_neural_predictability(
        frame, neural.Path("unused.png"), run_label="test", selected_epoch=0,
        horizon_trends=True,
    )
    assert len(captured["lines"]) == 2
    assert all(len(line.get_xdata()) == 2 for line in captured["lines"])
    assert captured["group_lines"] == 2
    _, trend = neural.horizon_linear_trend(frame.loc[frame["horizon_days"].eq(30)])
    assert trend["fit_available"]
    assert trend["auc_change_per_10x_events"] > 0
    assert not plt.get_fignums()


def test_full_outcome_atlas_keeps_unavailable_cells(monkeypatch) -> None:
    frame = pd.DataFrame({
        "outcome": ["a", "a", "b", "b"],
        "display_group": ["Clinical"] * 4,
        "horizon_days": [30, 365, 30, 365],
        "cif_auc": [.72, .81, float("nan"), .66],
        "tuning_primary_events": [2, 10, 1, 7],
    })
    captured = {}

    def capture(fig, path):
        captured["image"] = fig.axes[0].images[0].get_array().copy()
        captured["labels"] = [label.get_text() for label in fig.axes[0].get_yticklabels()]

    monkeypatch.setattr(neural, "save_fig", capture)
    neural.plot_neural_outcome_atlas(
        frame, neural.Path("unused.png"), groups={"Clinical": ["a", "b"]},
        run_label="test", selected_epoch=0,
    )
    assert captured["image"].shape == (2, 2)
    assert captured["image"].mask[1, 0]
    assert captured["labels"] == ["A", "B"]
    assert not plt.get_fignums()


def test_filter_panel_candidates_recomputes_display_subset(monkeypatch) -> None:
    decisions = pd.DataFrame({
        "outcome": ["a", "b", "c"],
        "disposition": ["primary_candidate", "definition_hold", "primary_candidate"],
    })
    monkeypatch.setattr(neural.pd, "read_csv", lambda _: decisions)
    frame = pd.DataFrame({"outcome": ["a", "a", "b", "c"], "cif_auc": [0.7, 0.8, 0.9, 0.6]})
    filtered = neural.filter_panel_candidates(frame, neural.Path("unused.csv"), expected_count=2)
    assert filtered["outcome"].tolist() == ["a", "a", "c"]
    with pytest.raises(ValueError, match="Expected 3 candidates"):
        neural.filter_panel_candidates(frame, neural.Path("unused.csv"), expected_count=3)
    decisions = pd.DataFrame({
        "outcome": ["a", "a"], "disposition": ["primary_candidate"] * 2,
    })
    with pytest.raises(ValueError, match="duplicate outcomes"):
        neural.filter_panel_candidates(frame, neural.Path("unused.csv"))


def test_candidate_display_groups_cover_exactly_60() -> None:
    import yaml

    groups = yaml.safe_load(neural.DEFAULT_CANDIDATE_GROUPS.read_text(encoding="utf-8"))
    members = [outcome for outcomes in groups.values() for outcome in outcomes]
    assert len(groups) == 12
    assert len(members) == len(set(members)) == 60
    assert min(map(len, groups.values())) >= 3
    assert max(map(len, groups.values())) <= 8
    frame = pd.DataFrame({"outcome": members, "display_group": ["old"] * 60})
    remapped = neural.apply_display_groups(frame, neural.DEFAULT_CANDIDATE_GROUPS)
    assert remapped["display_group"].nunique() == 12
    with pytest.raises(ValueError, match="exactly once"):
        neural.apply_display_groups(frame.iloc[:-1], neural.DEFAULT_CANDIDATE_GROUPS)
