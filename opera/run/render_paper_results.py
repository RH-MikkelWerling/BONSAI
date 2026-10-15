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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
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
    if missing:
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
        for key, source in {**plate.get("tables", {}), **plate.get("json_inputs", {})}.items():
            expanded = os.path.expandvars(source)
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in input path for {name}/{key}")
            path = Path(expanded)
            if not path.is_absolute():
                path = args.config.resolve().parent / path
            hashes[str(path)] = file_hash(path)
            inputs[key] = (json.loads(path.read_text()) if key in plate.get("json_inputs", {})
                           else read_table(path))
        inputs["mark"] = None
        inspect.signature(function).bind(**inputs)
        prepared.append((name, function, inputs, hashes))
    if args.validate_render:
        errors = {}
        for name, function, inputs, hashes in prepared:
            try:
                fig = function(**inputs)
                plt.close(fig)
            except Exception as exc:
                errors[name] = str(exc)
            finally:
                plt.close("all")
        if errors:
            raise ValueError("Figure schema/render validation failed: " + json.dumps(errors))
    if not args.execute:
        print(json.dumps({"figures": sorted(ids), "preview": True}))
        return
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    records = []
    for name, function, inputs, hashes in prepared:
        fig = function(**inputs)
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
        json.dumps({"recipe_sha": file_hash(args.config), "figures": records}, indent=2)
    )


if __name__ == "__main__":
    main()
