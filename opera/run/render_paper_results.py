"""Render real paper plates from explicit, source-hashed analysis tables.

No synthetic inputs are generated. Tables must satisfy the selected plotting
function's schema; recipe previews check paths and signatures before execution.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import yaml
from opera.run.analyse_locked_predictions import file_hash, read_table


def resolve_input(source, config):
    """Input path with environment variables expanded; relative to the recipe."""
    expanded = os.path.expandvars(source)
    if "$" in expanded:
        return None
    path = Path(expanded)
    return path if path.is_absolute() else (config.resolve().parent / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument('--report', type=Path,
                        help='Write an aggregate schema/render validation report (no patient rows)')
    parser.add_argument("--available-only", action="store_true",
                        help="Preview: render the figures whose inputs exist, list the rest")
    parser.add_argument("--validate-render", action="store_true",
                        help="Render every plate in memory to validate schemas without writing outputs")
    args = parser.parse_args()
    from opera.visualization import paper_figures as figures
    from opera.visualization.style import save_fig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recipe = yaml.safe_load(args.config.read_text())
    plates = recipe.get("figures", [])
    if not plates:
        raise ValueError("Register analysis tables for the real figure plates")
    prepared = []
    required = set(recipe.get("required_figures", []))
    missing = required - {p["id"] for p in plates}
    skipped = {}
    skipped.update({name: 'no registered recipe' for name in missing})
    if missing and not args.available_only:
        raise ValueError(f"Missing required figure recipes: {sorted(missing)}")
    ids = set()
    for plate in plates:
        name = plate["id"]
        if name in ids or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Figure IDs must be unique safe filenames")
        ids.add(name)
        kind = plate["function"]
        if not kind.startswith("figure_") or not callable(getattr(figures, kind, None)):
            raise ValueError("Unknown paper figure function")
        function = getattr(figures, kind)
        inputs = dict(plate.get("options", {}))
        hashes = {}
        if args.available_only:
            absent = [key for key, source in {**plate.get("tables", {}),
                                              **plate.get("json_inputs", {})}.items()
                      if (path := resolve_input(source, args.config)) is None
                      or not path.is_file()]
            if absent:
                skipped[name] = "missing inputs: " + ", ".join(absent)
                continue
        for key, source in {**plate.get("tables", {}), **plate.get("json_inputs", {})}.items():
            path = resolve_input(source, args.config)
            if path is None:
                raise ValueError(f"Unresolved environment variable in input path for {name}/{key}")
            hashes[str(path)] = file_hash(path)
            inputs[key] = (json.loads(path.read_text()) if key in plate.get("json_inputs", {})
                           else read_table(path))
        # e.g. OPERA_PAPER_FIGURE_MARK="TUNING PREVIEW" for figures of the tuning split.
        inputs["mark"] = os.environ.get("OPERA_PAPER_FIGURE_MARK") or None
        inspect.signature(function).bind(**inputs)
        prepared.append((name, function, inputs, hashes))
    code_hashes = {label: file_hash(Path(path)) for label, path in {
        'renderer': __file__, 'figures': inspect.getsourcefile(figures),
        'style': inspect.getsourcefile(save_fig)}.items()}
    validation = []
    if args.validate_render:
        errors = {}
        for name, function, inputs, hashes in prepared:
            try:
                fig = function(**inputs)
                fig.canvas.draw()
                warnings = []
                for key, value in inputs.items():
                    if hasattr(value, 'columns') and 'interval_status' in value.columns:
                        bad = value.interval_status.eq('insufficient_draws')
                        if bad.any():
                            warnings.append(f'{key}: {int(bad.sum())} rows have insufficient bootstrap draws')
                validation.append({'id': name, 'status': 'rendered', 'warnings': warnings,
                                   'input_hashes': hashes})
                plt.close(fig)
            except Exception as exc:
                errors[name] = str(exc)
                validation.append({'id': name, 'status': 'failed', 'error': str(exc)})
            finally:
                plt.close("all")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({'recipe_sha': file_hash(args.config),
                'code_sha256': code_hashes, 'figures': validation, 'skipped': skipped}, indent=2))
        if errors:
            raise ValueError("Figure schema/render validation failed: " + json.dumps(errors))
    if not args.execute:
        print(json.dumps({"figures": [p[0] for p in prepared], 'skipped': skipped,
                          'validated_render': args.validate_render, "preview": True}))
        return
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    records = []
    for name, function, inputs, hashes in prepared:
        try:
            fig = function(**inputs)
        except Exception as exc:
            if not args.available_only:
                raise
            skipped[name] = f"render failed: {exc}"
            plt.close("all")
            continue
        save_fig(fig, str(args.output_dir / (name + ".png")))
        plt.close(fig)
        records.append(
            {
                "id": name,
                "function": function.__name__,
                "input_hashes": hashes,
                "output_pdf_sha": file_hash(args.output_dir / (name + ".pdf")),
            }
        )
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps({"recipe_sha": file_hash(args.config), 'code_sha256': code_hashes,
                    'mark': os.environ.get('OPERA_PAPER_FIGURE_MARK'), "figures": records,
                    "skipped": skipped}, indent=2)
    )
    if skipped:
        print(json.dumps({"rendered": [r["id"] for r in records], "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
