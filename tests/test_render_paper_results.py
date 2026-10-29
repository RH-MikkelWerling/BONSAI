import json
import sys
import pandas as pd
import yaml
from opera.run.render_paper_results import main


def test_real_renderer_tracks_inputs_and_outputs(tmp_path, monkeypatch):
    from opera.visualization import paper_figures
    import matplotlib.pyplot as plt

    def figure_fixture(table, *, mark=None):
        assert mark is None
        fig, ax = plt.subplots()
        ax.plot(table["x"], table["y"])
        return fig

    monkeypatch.setattr(paper_figures, "figure_fixture", figure_fixture, raising=False)
    pd.DataFrame({"x": [0, 1], "y": [0.2, 0.4]}).to_csv(
        tmp_path / "input.csv", index=False
    )
    config = tmp_path / "recipe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "figures": [
                    {
                        "id": "test_plate",
                        "function": "figure_fixture",
                        "tables": {"table": "input.csv"},
                    }
                ]
            }
        )
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["render", "--config", str(config), "--output-dir", str(output), "--execute"],
    )
    main()
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert manifest["figures"][0]["input_hashes"]
    assert len(manifest["figures"][0]["output_pdf_sha"]) == 64
    assert (output / "test_plate.pdf").is_file()


def test_renderer_expands_server_paths_and_checks_required_plates(tmp_path, monkeypatch):
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "input.csv", index=False)
    config = tmp_path / "recipe.yaml"
    config.write_text(yaml.safe_dump({"required_figures": ["missing"], "figures": [
        {"id": "plate", "function": "figure_atlas", "tables": {"atlas": "${PLOT_ROOT}/input.csv"}}]}))
    monkeypatch.setenv("PLOT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["render", "--config", str(config), "--output-dir", str(tmp_path / "out")])
    import pytest
    with pytest.raises(ValueError, match="Missing required figure"):
        main()
    assert not (tmp_path / "out").exists()


def test_age_plate_separates_counts_predictions_and_observations():
    from opera.run.render_paper_figure_prototypes import synthetic_risk_patients
    from opera.visualization.paper_figures import figure_clinical_score_age
    import numpy as np
    import matplotlib.pyplot as plt
    patients = synthetic_risk_patients(np.random.default_rng(22), n=200)
    fig = figure_clinical_score_age(patients, outcome_label="Treatment failure", event_free=True)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        assert len(panels) == 6
        assert panels[0].images[0].get_array().sum() == patients.ipi_group.notna().sum()
        assert panels[1].images[0].get_array().sum() == patients.ipi_group.notna().sum()
        assert "predicted" in panels[2].get_title(loc="left")
        assert "Observed" in panels[4].get_title(loc="left")
    finally:
        plt.close(fig)


def test_server_workspace_registers_every_plate_with_absolute_paths(tmp_path, monkeypatch):
    from opera.run.prepare_paper_figure_inputs import main as prepare
    root = tmp_path / "server_figures"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(root), "--execute"])
    prepare()
    recipe = yaml.safe_load((root / "figures.yaml").read_text())
    assert {'supp_model_overview', 'supp_patient_disease_groups',
            'supp_router_arrows'} <= set(recipe['required_figures'])
    assert set(recipe["required_figures"]) == {p["id"] for p in recipe["figures"]}
    for plate in recipe["figures"]:
        for section in ("tables", "json_inputs"):
            for source in plate.get(section, {}).values():
                # Plot tables live in the workspace; outputs of other steps (the
                # clinical-score headline table) stay where those steps write them.
                assert (str(root / "plot_tables") in source
                        or source == str(tmp_path / "clinical_scores" / "headline_scores.csv")
                        or str(tmp_path / "source_removal") in source)
                assert "$" not in source.replace("${OPERA_PAPER_PLOT_SPLIT}", "tuning")
    assert not list((root / "plot_tables").iterdir())


def test_execution_records_save_failure_and_keeps_other_plates(tmp_path, monkeypatch):
    import pytest
    import matplotlib.pyplot as plt
    from opera.visualization import paper_figures, style
    def figure_failure_fixture(mark=None):
        return plt.subplots()[0]
    monkeypatch.setattr(paper_figures, 'figure_failure_fixture', figure_failure_fixture, raising=False)
    original = style.save_fig
    def save(fig, path):
        if Path(path).stem == 'bad':
            Path(path).write_text('partial')
            raise RuntimeError('simulated disk save failure')
        return original(fig, path)
    from pathlib import Path
    monkeypatch.setattr(style, 'save_fig', save)
    config = tmp_path / 'recipe.yaml'
    config.write_text(yaml.safe_dump({'figures': [dict(id=name, function='figure_failure_fixture')
                                                for name in ('bad', 'good')]}))
    out = tmp_path / 'figures'
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['render', '--config', str(config), '--output-dir', str(out),
                                     '--available-only', '--execute', '--report', str(report)])
    with pytest.raises(ValueError, match='simulated disk save failure'):
        main()
    manifest = json.loads(report.read_text())
    assert 'bad' in manifest['render_failures'] and not manifest['complete']
    assert [r['id'] for r in manifest['figures']] == ['good']
    assert not (out / 'bad.png').exists()
    assert (out / 'good.pdf').exists()
