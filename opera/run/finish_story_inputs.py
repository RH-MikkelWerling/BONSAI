"""Server step for the story version of the OPERA paper (16 Nov 2026).

Runs alongside ``opera.run.finish_figure_inputs`` (Codex's recover/maps/arrows lanes) and
covers what those lanes do not:

  embeddings   encoder comparison plates (joint and OPERA; cohort, national, adapted encoders)
               with ``figure_encoder_embeddings``: disease group, computed IPI in its four
               standard categories, observed one-year overall-survival risk relative to the
               average. Coordinates stay in ``server_only/``.
  clinical     the held-out adapted-OPERA clinical plates re-rendered in the thesis style
               (same groups; fixed wording), with figures and aggregate tables. FLIPI2 is
               evaluated against one-year overall survival (``--flipi2-endpoint``): the
               next-treatment-or-death proxy was not interpretable in follicular lymphoma
               (decided 17 Nov 2026, after the held-out results; reported as such).
  no_index     held-out patients without any recorded prognostic index (counts, small cells
               suppressed).

No training, no inference on new patients, no change to analysis choices. Without
``--execute`` it only checks inputs and prints the plan. Each step records its status; a
failed step does not stop the others. Everything to download ends up in one file:

    <output-root>/download_story.zip

Typical use, from the server repository root:

    python -m opera.run.finish_story_inputs \
        --national data/paper_256_v3_national_20261111 \
        --inventory interpretation_inventory_20261115.json \
        --patient-metadata data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet \
        --clinical-workspace data/paper_256_v3_national_20261111/clinical_20261115/R_EA
    # check the plan, then add --execute
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

IPI_FROM_SCORE = {0: "Low", 1: "Low", 2: "Low-intermediate", 3: "High-intermediate", 4: "High", 5: "High"}
LABELS = {"Local joint": "Cohort encoder", "National joint": "National encoder", "Adapted joint": "Adapted encoder",
          "Local OPERA": "Cohort encoder", "National OPERA": "National encoder", "Adapted OPERA": "Adapted encoder"}
SUPPRESS = 5


def _render_script(repo: Path):
    path = repo / "scripts/render_recovered_encoder_embeddings.py"
    spec = importlib.util.spec_from_file_location("render_recovered_encoder_embeddings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _survival_paths(national: Path):
    # Several adapted architectures share this directory tree. Bind to R_EA,
    # rather than requiring the whole EA tree to contain exactly one export.
    bundle = json.loads((national / 'adapted_fits/bundle.json').read_text())
    row, = [r for r in bundle['runs'] if r['name'] == 'seed42/EA/R']
    path = national / 'predictions' / row['run_directory'] / 'held_out/direct_cr_predictions.parquet'
    return [path] if path.is_file() else []


def _survival(path: Path):
    import pandas as pd
    from opera.run.build_paper_embedding_tables import _patient_ids
    frame = _patient_ids(pd.read_parquet(path, columns=["subject_id", "outcome", "time_days", "event"]))
    frame = frame.loc[frame.outcome.eq("overall_survival"), ["subject_id", "time_days", "event"]].drop_duplicates()
    if frame.subject_id.duplicated().any():
        raise ValueError("More than one overall-survival row per patient")
    return frame.rename(columns={"subject_id": "patient_id"})


def _score_config(workspace: Path, explicit: Path | None):
    import yaml
    if explicit:
        return explicit
    # launch_adapted_clinical_comparison uses the flat workspace layout.
    canonical = workspace / 'clinical_scores.yaml'
    if canonical.is_file():
        return canonical
    found = [p for p in sorted((workspace / "configs").glob("*.yaml"))
             if "tuning" not in p.name and "scores" in (yaml.safe_load(p.read_text()) or {})]
    if len(found) != 1:
        raise ValueError(f"Pass --score-config; held-out score configs found: {[p.name for p in found]}")
    return found[0]


def _clinical_analysis(workspace):
    for path in (workspace / 'analysis.yaml', workspace / 'analysis_inventory/analysis.yaml'):
        if path.is_file():
            return path
    raise FileNotFoundError(f'No clinical analysis.yaml in {workspace}')


def flipi2_os_config(config):
    """Patch either a template or an already expanded frozen clinical recipe.

    prepare_clinical_registry_refresh expands native endpoints into scores and
    deliberately clears native_endpoints. Do not regenerate other index jobs.
    """
    result = copy.deepcopy(config)
    patch = dict(
        outcome='overall_survival', outcome_label='Overall survival (original secondary endpoint)',
        endpoint_match='original_secondary_endpoint', death_in_endpoint=True,
        original_endpoint='overall survival (secondary endpoint; principal endpoint was progression-free survival)',
        horizons_days=[365], horizon_days=365,
        endpoint_proxy_note=('OS was examined in the original FLIPI2 study; its principal endpoint, '
                             'PFS, could not be approximated: the next-treatment-or-death proxy was '
                             'not interpretable in follicular lymphoma. Endpoint changed after the '
                             'held-out results, 17 Nov 2026.'))
    def is_flipi2(value):
        return ''.join(c for c in str(value).casefold() if c.isalnum()) == 'flipi2'
    matched = 0
    for key, entry in (result.get('native_endpoints') or {}).items():
        if is_flipi2(key):
            entry.update(patch)
            matched += 1
    for entry in result.get('scores', []):
        values = entry.get('index_values') or []
        if isinstance(values, str):
            values = [values]
        if any(is_flipi2(v) for v in [*values, entry.get('score_label', '')]):
            entry.update(patch)
            matched += 1
    if not matched:
        raise ValueError('No explicitly identified FLIPI2 entry in scores or native_endpoints; '
                         'check the frozen clinical_scores.yaml')
    return result


def refresh_discrimination(analysis_path, score_path, output):
    """Use the same expanded, endpoint-patched recipe as the clinical plates."""
    import yaml
    from opera.run.analyse_locked_predictions import read_table
    from opera.run.render_clinical_score_panels import expand_index_scores
    from opera.run.launch_adapted_clinical_comparison import discrimination
    analysis = yaml.safe_load(Path(analysis_path).read_text())
    metadata = read_table(analysis['patient_metadata'])
    metadata.subject_id = metadata.subject_id.astype(str)
    metadata = metadata.rename(columns=analysis.get('patient_metadata_column_map', {}))
    specs = expand_index_scores(yaml.safe_load(Path(score_path).read_text()), metadata)
    if not specs:
        raise ValueError('No expanded clinical scores for discrimination')
    discrimination(analysis, specs, metadata, Path(output))


class Run:
    def __init__(self, root: Path, execute: bool, redo=()):
        self.root, self.execute, self.redo = root, execute, set(redo or ())
        self.ready = root / "ready"
        self.private = root / "server_only"
        self.status_path = root / "story_status.json"
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}

    def step(self, name, check, action):
        try:
            plan = check()
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            self.status[name] = {"status": "blocked", "reason": str(error)}
            print(f"[{name}] blocked: {error}")
            return
        if not self.execute:
            print(f"[{name}] ready to run: {json.dumps(plan, default=str)}")
            return
        if self.status.get(name, {}).get("status") == "complete" and name not in self.redo:
            print(f"[{name}] already complete")
            return
        try:
            result = action()
            self.status[name] = {"status": "complete", **(result or {})}
            print(f"[{name}] complete")
        except Exception as error:  # noqa: BLE001
            self.status[name] = {"status": "failed", "reason": str(error), "trace": traceback.format_exc()[-2000:]}
            print(f"[{name}] failed: {error}")
        self.status_path.write_text(json.dumps(self.status, indent=2, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--national", type=Path, default=Path("data/paper_256_v3_national_20261111"))
    p.add_argument("--inventory", type=Path, default=Path("interpretation_inventory_20261115.json"))
    p.add_argument("--patient-metadata", type=Path,
                   default=Path("data/clinical_registry_refresh/20261111/prepared_clinical/patient_metadata.parquet"))
    p.add_argument("--survival-predictions", type=Path,
                   help="Held-out R_EA direct_cr_predictions.parquet (default: the one under --national)")
    p.add_argument("--clinical-workspace", type=Path, help="Completed held-out R_EA clinical workspace")
    p.add_argument("--score-config", type=Path, help="Its held-out score configuration, if not found automatically")
    p.add_argument("--output-root", type=Path, default=Path("data/story_closure_20261116"))
    p.add_argument("--only", nargs="*", choices=["embeddings", "clinical", "no_index"])
    p.add_argument("--flipi2-endpoint", choices=["overall_survival", "treatment_failure"],
                   default="overall_survival", help="Endpoint for the FLIPI2 comparison (see module docstring)")
    p.add_argument("--redo", nargs="*", default=[], choices=["embeddings", "clinical", "no_index"],
                   help="Rerun these steps even if an earlier run completed them")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    repo = Path.cwd()
    os.environ.update(MPLBACKEND="Agg", OPERA_THESIS_STYLE="1", OPERA_PAPER_FIGURE_MARK="",
                      CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    run = Run(a.output_root.resolve(), a.execute, a.redo)
    if a.execute:
        for d in (run.ready / "figures", run.ready / "tables", run.private):
            d.mkdir(parents=True, exist_ok=True)
    wanted = set(a.only or ["embeddings", "clinical", "no_index"])

    def survival_path():
        if a.survival_predictions:
            return a.survival_predictions
        found = _survival_paths(a.national)
        if len(found) != 1:
            raise ValueError(f"Pass --survival-predictions; candidates under --national: {[str(x) for x in found]}")
        return found[0]

    # -- embeddings ------------------------------------------------------------------------
    if "embeddings" in wanted:
        def check():
            import json as _j
            for path in (a.inventory, a.patient_metadata):
                if not path.is_file():
                    raise FileNotFoundError(path)
            stages = _render_script(repo).select_stages(_j.loads(a.inventory.read_text()))
            return {"stages": stages, "survival": str(survival_path())}

        def action():
            import matplotlib.pyplot as plt
            import matplotlib.text
            import pandas as pd
            import yaml
            from opera.run.analyse_locked_predictions import read_table
            from opera.run.build_paper_embedding_tables import _patient_ids, embedding_comparison
            from opera.run.paper_story_figures import thesis_style
            from opera.visualization.encoder_embedding_comparison import figure_encoder_embeddings
            from opera.visualization.style import setup_style
            setup_style()
            thesis_style()
            stages = _render_script(repo).select_stages(json.loads(a.inventory.read_text()))
            metadata = _patient_ids(read_table(a.patient_metadata))
            if "ipi_computed_score" not in metadata:
                raise ValueError("patient metadata lacks ipi_computed_score (use the prepared clinical metadata)")
            score = metadata.set_index("subject_id").ipi_computed_score
            group_map = yaml.safe_load(Path("opera/configs/paper_embedding_tables.example.yaml").read_text()).get("ipi_group_map")
            survival = _survival(survival_path())
            out = {}
            for family, family_stages in stages.items():
                coords, _ = embedding_comparison(family_stages, metadata, method="umap", group_map=group_map)
                values = pd.to_numeric(coords.patient_id.map(score), errors="coerce")
                coords["ipi_group"] = values.map(lambda v: IPI_FROM_SCORE.get(int(v)) if pd.notna(v) else None)
                coords = coords.loc[coords.patient_id.isin(set(survival.patient_id))]
                coords.to_csv(run.private / f"{family}_encoder_coordinates.csv", index=False)
                from opera.visualization.encoder_embedding_comparison import neighbourhood_metrics
                metrics = neighbourhood_metrics(coords, survival)
                metrics.to_csv(run.ready / "tables" / f"embedding_neighbourhood_{family}.csv", index=False)
                fig = figure_encoder_embeddings(coords, survival, model_order=list(family_stages), model_labels=LABELS,
                                                metrics=metrics)
                for t in fig.findobj(matplotlib.text.Text):
                    t.set_fontfamily("serif")
                stem = run.ready / "figures" / f"supp_embedding_comparison_{family}"
                fig.savefig(stem.with_suffix(".pdf"))
                fig.savefig(stem.with_suffix(".png"), dpi=200)
                plt.close(fig)
                out[family] = {"patients": int(coords.patient_id.nunique()),
                               "with_ipi": int(coords.drop_duplicates("patient_id").ipi_group.notna().sum()),
                               "projection": sorted(coords.projection.unique()) if "projection" in coords else None}
            return out
        run.step("embeddings", check, action)

    # -- clinical plates ----------------------------------------------------------------------
    if "clinical" in wanted:
        def check():
            if not a.clinical_workspace:
                raise ValueError("Pass --clinical-workspace (the completed held-out R_EA clinical workspace)")
            analysis = _clinical_analysis(a.clinical_workspace)
            return {"analysis_config": str(analysis), "score_config": str(_score_config(a.clinical_workspace, a.score_config))}

        def action():
            import yaml
            from opera.run.refresh_clinical_figures import package_clinical
            plan = check()
            target = run.private / "clinical_scores"
            if target.exists():
                shutil.rmtree(target)
            score_config = Path(plan["score_config"])
            if a.flipi2_endpoint == "overall_survival":
                config = flipi2_os_config(yaml.safe_load(score_config.read_text()))
                score_config = run.private / "score_config_flipi2_os.yaml"
                score_config.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run([sys.executable, "-m", "opera.run.render_clinical_score_panels",
                            "--analysis-config", plan["analysis_config"], "--score-config", str(score_config),
                            "--output-dir", str(target), "--split", "held_out", "--execute"],
                           env=dict(os.environ), check=True)
            refresh_discrimination(plan['analysis_config'], score_config,
                                   target / 'clinical_discrimination.csv')
            archive = run.ready / "clinical_R_EA_thesis_style.zip"
            archive.unlink(missing_ok=True)
            package_clinical(target, archive)
            return {"archive": str(archive), "flipi2_endpoint": a.flipi2_endpoint}
        run.step("clinical", check, action)

    # -- patients without a recorded index ----------------------------------------------------
    if "no_index" in wanted:
        def check():
            if not a.patient_metadata.is_file():
                raise FileNotFoundError(a.patient_metadata)
            if not a.clinical_workspace:
                raise ValueError("Pass --clinical-workspace (its score configuration names the recorded indices)")
            return {"metadata": str(a.patient_metadata), "survival": str(survival_path()),
                    "score_config": str(_score_config(a.clinical_workspace, a.score_config))}

        def action():
            import pandas as pd
            import yaml
            from opera.run.analyse_locked_predictions import read_table
            from opera.run.build_paper_embedding_tables import _patient_ids
            config = yaml.safe_load(_score_config(a.clinical_workspace, a.score_config).read_text())
            specs = list(config.get("scores", []))
            columns = sorted({s["group_column"] for s in specs if s.get("group_column")
                              and not str(s.get("id", "")).startswith(("ipi_computed", "nccn"))})
            metadata = _patient_ids(read_table(a.patient_metadata))
            missing = [c for c in columns if c not in metadata]
            if missing:
                raise ValueError(f"Recorded-index columns not in metadata: {missing}")
            held = set(_survival(survival_path()).patient_id)
            data = metadata.loc[metadata.subject_id.isin(held)].drop_duplicates("subject_id")
            none = data[columns].isna().all(axis=1)
            by = data.assign(no_index=none).groupby("cohort_grouped").agg(patients=("no_index", "size"),
                                                                           no_index=("no_index", "sum")).reset_index()
            for col in ("patients", "no_index"):
                by[col] = by[col].where(by[col] >= SUPPRESS)  # small counts suppressed
            by.to_csv(run.ready / "tables/no_index_by_disease.csv", index=False)
            summary = {"held_out_patients": int(len(data)), "without_recorded_index": int(none.sum()),
                       "share": round(float(none.mean()), 4), "index_columns": columns}
            (run.ready / "tables/no_index_summary.json").write_text(json.dumps(summary, indent=2))
            return summary
        run.step("no_index", check, action)

    # -- package -----------------------------------------------------------------------------
    if a.execute:
        run.status_path.write_text(json.dumps(run.status, indent=2, default=str))
        archive = run.root / "download_story.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
            for path in sorted(run.ready.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(run.ready))
            z.write(run.status_path, run.status_path.name)
        print(f"Download only: {archive}")
    print(json.dumps({k: v.get("status") for k, v in run.status.items()}, indent=2))
    if a.execute and any(run.status.get(name, {}).get('status') != 'complete' for name in wanted):
        raise SystemExit('Story export incomplete; inspect story_status.json (partial archive retained)')


if __name__ == "__main__":
    main()
