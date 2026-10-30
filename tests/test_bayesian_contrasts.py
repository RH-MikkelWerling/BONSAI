"""Scientific algebra, patient covariance, fixed rosters and Bayesian artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opera.analysis.bayesian_contrasts import (
    FAMILIES,
    build_contrast_model,
    contrast_design,
    fit_contrast_model,
    panel_weights,
    prepare_contrast_data,
    summarize_draws,
)


def fixture(family="pooling", seeds=(42, 43, 44)):
    rng = np.random.default_rng(7)
    basis, _, _ = contrast_design(family)
    rows, index, draws = [], [], []
    shared = rng.normal(0, 0.01, (2, 150))
    for cohort_number, cohort in enumerate(("A", "B")):
        for outcome_number, outcome in enumerate(("u", "v", "w", "x")):
            for contrast_number, contrast in enumerate(basis):
                value = 0.01 * (contrast_number + 1) + 0.001 * outcome_number
                sampled = value + shared[cohort_number] + rng.normal(0, 0.003, 150)
                for seed in seeds:
                    identity = dict(
                        split="held_out",
                        seed=seed,
                        cohort=cohort,
                        outcome=outcome,
                        horizon_days=365,
                        contrast=contrast,
                        metric="auc",
                    )
                    rows.append(
                        dict(
                            **identity,
                            gain=value,
                            n_events_train=2 ** (outcome_number + cohort_number),
                            endpoint_group="grade_pair"
                            if outcome in ("u", "v")
                            else outcome,
                            cohort_group="all",
                            outcome_family="family",
                            n_cases=40,
                            n_controls=60,
                        )
                    )
                    index.append(identity)
                    # Identical fitted seeds must not artificially reduce patient SE.
                    draws.append(sampled)
    return pd.DataFrame(rows), pd.DataFrame(index), np.stack(draws)


def prepare(*args, **kwargs):
    return prepare_contrast_data(*args, family="pooling", metric="auc", **kwargs)


def test_pooling_and_external_identities_hold_for_every_posterior_draw():
    for family in FAMILIES:
        basis, contrasts, matrix = contrast_design(family)
        assert np.linalg.matrix_rank(matrix) == len(basis)
        effects = np.random.default_rng(3).normal(size=(20, len(basis))) @ matrix.T

        def get(name):
            return effects[:, contrasts.index(name)]

        if family == "pooling":
            np.testing.assert_allclose(get("P5"), get("P1") + get("P4"))
            np.testing.assert_allclose(get("P5"), get("P2") + get("P3"))
            np.testing.assert_allclose(get("P6"), get("P3") - get("P1"))
        elif family == "external":
            np.testing.assert_allclose(get("E3"), get("E1") + get("A1"))
            np.testing.assert_allclose(get("E4"), get("E2") + get("A2"))
            np.testing.assert_allclose(get("I3"), get("I1") + get("I2"))


def test_bootstrap_covariance_preserves_seed_and_endpoint_dependence():
    data = prepare(*fixture())
    one_seed = prepare(*fixture(seeds=(42,)), seeds=(42,))
    np.testing.assert_allclose(data.bootstrap, one_seed.bootstrap)
    np.testing.assert_allclose(
        data.blocks[0]["covariance"], one_seed.blocks[0]["covariance"]
    )
    covariance = data.blocks[0]["covariance"]
    assert covariance[0, 3] > 0
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert len(data.blocks) == 2
    np.testing.assert_allclose(data.weights @ data.design, 0, atol=1e-12)


def test_exact_index_alignment_survives_shuffling():
    effects, index, gains = fixture()
    order = np.random.default_rng(9).permutation(len(index))
    shuffled = prepare(
        effects.sample(frac=1, random_state=4), index.iloc[order], gains[order]
    )
    expected = prepare(effects, index, gains)
    np.testing.assert_allclose(shuffled.observed, expected.observed)
    np.testing.assert_allclose(shuffled.bootstrap, expected.bootstrap)


@pytest.mark.parametrize(
    "change",
    [
        "missing_seed",
        "duplicate",
        "negative_support",
        "changed_support",
        "missing_draw",
        "nan_gain",
        "invalid_group",
    ],
)
def test_invalid_rosters_or_metadata_fail_closed(change):
    effects, index, gains = fixture()
    if change == "missing_seed":
        effects = effects.iloc[1:]
    elif change == "duplicate":
        effects = pd.concat([effects, effects.iloc[:1]])
    elif change == "negative_support":
        effects.loc[0, "n_events_train"] = -1
    elif change == "changed_support":
        effects.loc[0, "n_events_train"] = 999
    elif change == "missing_draw":
        index, gains = index.iloc[1:], gains[1:]
    elif change == "nan_gain":
        effects.loc[0, "gain"] = np.nan
    elif change == "invalid_group":
        effects.loc[0, "endpoint_group"] = "another"
    with pytest.raises(ValueError):
        prepare(effects, index, gains)


def test_zero_training_events_are_retained_and_trend_is_explicit():
    effects, index, gains = fixture()
    effects.loc[effects.outcome.eq("u"), "n_events_train"] = 0
    data = prepare(effects, index, gains)
    assert (data.cells.n_events_train == 0).sum() == 2
    assert np.isfinite(data.design).all()
    assert prepare(effects, index, gains, trend="none").design.sum() == 0
    spline = prepare(effects, index, gains, trend="spline", spline_knots=4)
    assert spline.design.shape[1] > 1


def test_weighting_is_equal_disease_then_endpoint_group_then_member():
    cells = pd.DataFrame(
        {
            "cohort": ["A", "A", "A", "B"],
            "outcome": ["u", "v", "w", "u"],
            "endpoint_group": ["pair", "pair", "other", "pair"],
        }
    )
    np.testing.assert_allclose(
        panel_weights(cells, "equal_disease_endpoint_group"), [0.125, 0.125, 0.25, 0.5]
    )
    with pytest.raises(ValueError):
        panel_weights(
            cells.drop(columns="endpoint_group"), "equal_disease_endpoint_group"
        )


def test_insufficient_joint_draws_do_not_use_pairwise_covariance():
    effects, index, gains = fixture()
    gains[::3, :40] = np.nan
    with pytest.raises(ValueError, match="jointly finite"):
        prepare(effects, index, gains)


def test_probabilities_distinguish_benefit_harm_and_equivalence():
    summary = summarize_draws(np.array([-0.02, -0.005, 0.005, 0.02]), 0.01)
    assert summary["probability_benefit"] == 0.5
    assert summary["probability_meaningful_benefit"] == 0.25
    assert summary["probability_meaningful_harm"] == 0.25
    assert summary["probability_practical_equivalence"] == 0.5


def test_model_samples_and_saves_prior_checks_and_exact_contrast_algebra(tmp_path):
    pytest.importorskip("pymc")
    az = pytest.importorskip("arviz")
    data = prepare(*fixture())
    output = fit_contrast_model(
        data,
        output_dir=tmp_path / "fit",
        prior_scale=0.1,
        random_effect_scale=0.05,
        slope_scale=0.02,
        practical_tolerance=0.01,
        draws=6,
        tune=6,
        chains=1,
        cores=1,
        prior_draws=10,
        require_convergence=False,
        progressbar=False,
    )
    posterior = az.from_netcdf(output / "posterior.nc").posterior
    effect = posterior.cell_effect
    np.testing.assert_allclose(
        effect.sel(contrast="P5"), effect.sel(contrast="P1") + effect.sel(contrast="P4")
    )
    assert (output / "prior_predictive.nc").is_file()
    assert (output / "predictive_checks.csv").is_file()
    audit = json.loads((output / "model_specification.json").read_text())
    assert audit["status"] == "diagnostic_only_not_reportable"
    assert audit["nest_diseases"] is False
    assert audit["nest_outcomes"] is False
    from opera.visualization.bayesian_contrast_plots import plot_contrast_results

    for figure in plot_contrast_results(output):
        assert figure.is_file() and figure.stat().st_size > 1000
    panel = pd.read_csv(output / "posterior_panel.csv")
    assert set(panel.contrast) == set(data.contrasts)
    with pytest.raises(FileExistsError):
        fit_contrast_model(
            data,
            output_dir=output,
            prior_scale=0.1,
            random_effect_scale=0.05,
            slope_scale=0.02,
            practical_tolerance=0.01,
        )


def test_nested_sensitivity_requires_explicit_clinical_metadata():
    pytest.importorskip("pymc")
    effects, index, gains = fixture()
    data = prepare(effects.drop(columns="outcome_family"), index, gains)
    with pytest.raises(ValueError, match="outcome_family"):
        build_contrast_model(
            data,
            prior_scale=0.1,
            random_effect_scale=0.05,
            slope_scale=0.02,
            nest_outcomes=True,
        )


def test_prior_only_does_not_fit_a_posterior(tmp_path, monkeypatch):
    pm = pytest.importorskip("pymc")

    def forbidden(*args, **kwargs):
        raise AssertionError("Prior-only must not sample a posterior")

    monkeypatch.setattr(pm, "sample", forbidden)
    output = fit_contrast_model(
        prepare(*fixture()),
        output_dir=tmp_path / "prior",
        prior_scale=0.1,
        random_effect_scale=0.05,
        slope_scale=0.02,
        practical_tolerance=0.01,
        prior_draws=5,
        prior_only=True,
    )
    assert (output / "prior_panel.csv").is_file()
    assert not (output / "posterior.nc").exists()
    assert (
        json.loads((output / "model_specification.json").read_text())["status"]
        == "prior_only_no_posterior_fitted"
    )


def test_entirely_missing_family_cells_require_an_explicit_roster():
    effects, index, gains = fixture()
    extra = effects.iloc[:1].copy()
    extra["cohort"] = "C"
    extra["contrast"] = "R1"
    combined = pd.concat([effects, extra], ignore_index=True)
    with pytest.raises(ValueError, match="no required family contrasts"):
        prepare(combined, index, gains)
    roster = effects[["cohort", "outcome"]].drop_duplicates()
    data = prepare(combined, index, gains, cell_roster=roster)
    assert data.specification["available_cells"] == 9
    assert data.specification["declared_cells"] == 8


@pytest.mark.parametrize("family", list(FAMILIES))
def test_runner_validates_shared_draw_provenance_and_estimand(tmp_path, family):
    from opera.run.bayesian_contrasts import run
    from pathlib import Path
    import yaml

    canonical = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "opera/configs/manifests/final_contrasts_20261014.yaml"
        ).read_text()
    )
    expressions = {
        name: expr
        for rows in canonical["essential_contrasts"].values()
        for name, expr in rows.items()
    }

    effects, index, gains = fixture(family=family)
    source = tmp_path / "analysis"
    source.mkdir()
    effects.to_csv(source / "contrasts.csv", index=False)
    index.to_csv(source / "paired_bootstrap_index.csv", index=False)
    np.savez_compressed(source / "paired_bootstrap_draws_INTERNAL.npz", gains=gains)
    manifest = source / "analysis_manifest.json"
    manifest.write_text(json.dumps({"estimand": "binary_known_status_unweighted"}))
    config = {
        "analysis_dir": str(source),
        "output_dir": str(tmp_path / "fit"),
        "family": family,
        "metric": "auc",
        "seeds": [42, 43, 44],
        "estimand": "binary_known_status_unweighted",
        "prior_scale": 0.1,
        "random_effect_scale": 0.05,
        "slope_scale": 0.02,
        "practical_tolerance": 0.01,
        "validate_only": True,
    }
    with pytest.raises(ValueError, match="shared patient draws"):
        run(config)
    manifest.write_text(
        json.dumps(
            {
                "estimand": "binary_known_status_unweighted",
                "contrast_expressions": expressions,
                "shared_bootstrap_axes": [
                    "models",
                    "outcomes",
                    "horizons",
                    "training_seeds",
                ],
            }
        )
    )
    data = run(config)
    assert len(data.cells) == 8
    assert not (tmp_path / "fit").exists()
    payload = json.loads(manifest.read_text())
    payload["contrast_expressions"][next(iter(FAMILIES[family]))] = "gain(A,B)"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical contrast"):
        run(config)
    config["estimand"] = "IPCW"
    with pytest.raises(ValueError, match="estimand"):
        run(config)


def test_conditional_simulation_matches_analytic_gaussian_mean_and_covariance():
    from types import SimpleNamespace
    xr = pytest.importorskip('xarray')
    from opera.analysis.bayesian_contrasts import add_conditional_cell_draws

    data = prepare(*fixture())
    n_draws = 4000
    mean = np.full((1, n_draws, len(data.cells), len(data.basis)), 0.01)
    sigma = np.full((1, n_draws, len(data.basis)), 0.04)
    group = xr.Dataset(
        {
            "basis_mean": (("chain", "draw", "cell", "basis"), mean),
            "sigma_cell": (("chain", "draw", "basis"), sigma),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draws),
            "cell": np.arange(len(data.cells)),
            "basis": data.basis,
        },
    )
    result = add_conditional_cell_draws(SimpleNamespace(posterior=group), data)
    block = data.blocks[0]
    positions = block["positions"]
    samples = result.posterior.basis_effect.values[0][:, positions, :].reshape(
        n_draws, -1
    )
    d = np.eye(samples.shape[1]) * 0.04**2
    covariance = block["covariance"]
    expected_mean = 0.01 + d @ np.linalg.solve(
        d + covariance, data.observed[positions].reshape(-1) - 0.01
    )
    expected_covariance = d - d @ np.linalg.solve(d + covariance, d)
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.0005)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False), expected_covariance, atol=1e-5, rtol=0.12
    )


def sparse_analysis(root, *, draws=150):
    """Pooling-only analysis outputs with realistic sparse-panel defects."""
    rng = np.random.default_rng(11)
    basis, _, _ = contrast_design("pooling")
    rows, index, samples = [], [], []
    for cohort_number, cohort in enumerate(("A", "B", "C")):
        for outcome_number, outcome in enumerate(("u", "v", "w", "x")):
            single_class = (cohort, outcome) == ("A", "u")
            for metric in ("auc", "brier"):
                for contrast_number, contrast in enumerate(basis):
                    if (cohort, outcome, contrast) == ("C", "w", "P2"):
                        continue  # one model arm absent for this cell
                    for seed in (42, 43):
                        identity = dict(
                            split="held_out", seed=seed, cohort=cohort,
                            outcome=outcome, horizon_days=365,
                            contrast=contrast, metric=metric,
                        )
                        auc_missing = metric == "auc" and single_class
                        value = np.nan if auc_missing else 0.01 * (contrast_number + 1)
                        sampled = value + rng.normal(0, 0.003, draws)
                        if metric == "auc" and (cohort, outcome) == ("B", "v") and seed == 43:
                            sampled[: int(0.3 * draws)] = np.nan  # degenerate resamples
                        rows.append(
                            dict(
                                **identity, gain=value,
                                n_events_train=2 ** (outcome_number + cohort_number),
                                endpoint_group=outcome, cohort_group="all",
                                outcome_family="family",
                                n_cases=0 if single_class else 40, n_controls=60,
                            )
                        )
                        index.append(identity)
                        samples.append(sampled)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "contrasts.csv", index=False)
    pd.DataFrame(index).to_csv(root / "paired_bootstrap_index.csv", index=False)
    np.savez(root / "paired_bootstrap_draws_INTERNAL.npz", gains=np.stack(samples))
    return pd.DataFrame(rows), pd.DataFrame(index), np.stack(samples)


def test_estimable_roster_excludes_sparse_cells_with_reasons(tmp_path):
    from opera.analysis.bayesian_contrasts import estimable_cell_roster

    effects, index, gains = sparse_analysis(tmp_path)
    roster, excluded = estimable_cell_roster(
        effects, index, gains, family="pooling", metric="auc", seeds=(42, 43)
    )
    reasons = excluded.set_index(["cohort", "outcome"]).reason.to_dict()
    assert reasons == {
        ("A", "u"): "auc_evaluation_support_below_minimum_or_single_class",
        ("B", "v"): "insufficient_finite_bootstrap_draws",
        ("C", "w"): "missing_contrast_model_or_seed",
    }
    assert len(roster) == 9
    data = prepare_contrast_data(
        effects, index, gains, family="pooling", metric="auc",
        seeds=(42, 43), cell_roster=roster,
    )
    assert len(data.cells) == 9
    with pytest.raises(ValueError):
        prepare_contrast_data(
            effects, index, gains, family="pooling", metric="auc", seeds=(42, 43)
        )
    brier, brier_excluded = estimable_cell_roster(
        effects, index, gains, family="pooling", metric="brier", seeds=(42, 43)
    )
    # A single-class cell has no AUC but still has a proper probability score.
    assert ("A", "u") in set(map(tuple, brier.to_numpy()))
    assert set(brier_excluded.reason) == {"missing_contrast_model_or_seed"}


def test_recipes_validate_rosters_and_report_missing_families(tmp_path):
    import yaml
    from opera.run.prepare_paper_bayes_recipes import prepare as prepare_recipes

    sparse_analysis(tmp_path / "analysis")
    template = {
        "metric": "auc", "prior_scale": 0.05, "random_effect_scale": 0.02,
        "slope_scale": 0.01, "practical_tolerance": 0.01,
        "metric_settings": {"brier": {"prior_scale": 0.01}},
    }
    primary = prepare_recipes(
        template, tmp_path / "recipes", tmp_path / "analysis",
        tmp_path / "results", (42, 43),
    )
    names = sorted(Path(p).name for p in primary)
    assert names == ["pooling_auc_support_spline.yaml", "pooling_brier_support_spline.yaml"]
    auc = yaml.safe_load((tmp_path / "recipes" / names[0]).read_text())
    brier = yaml.safe_load((tmp_path / "recipes" / names[1]).read_text())
    assert auc["prior_scale"] == 0.05 and brier["prior_scale"] == 0.01
    assert brier["practical_tolerance"] is None  # never reuse an AUC-scale lock
    assert Path(auc["cell_roster"]).is_file()
    status = pd.read_csv(tmp_path / "recipes" / "recipe_status.csv")
    external = status.loc[status.family.eq("external")]
    assert external.status.eq("not_estimable").all()
    manifest = json.loads((tmp_path / "recipes" / "recipe_manifest.json").read_text())
    assert manifest["estimability_rule"]["depends_on_effect_size"] is False
    with pytest.raises(FileExistsError):
        prepare_recipes(
            template, tmp_path / "recipes", tmp_path / "analysis",
            tmp_path / "results", (42, 43),
        )


def test_support_bins_are_piecewise_constant_and_require_populated_bins():
    effects, index, gains = fixture()
    # Training events are 2 ** (outcome + cohort): {1, 2, 2} below 3, the rest above.
    data = prepare(effects, index, gains, trend="bins", support_bins=[0, 3])
    assert data.design.shape == (8, 2)
    np.testing.assert_allclose(data.weights @ data.design, 0, atol=1e-12)
    assert data.specification["transform"]["cells_per_bin"] == [3, 5]
    assert len(data.grid_support) == 2
    with pytest.raises(ValueError, match="at least 3 cells"):
        prepare(effects, index, gains, trend="bins", support_bins=[0, 3, 10])
    with pytest.raises(ValueError, match="starting at 0"):
        prepare(effects, index, gains, trend="bins", support_bins=[5, 3])
