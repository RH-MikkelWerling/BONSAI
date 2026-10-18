"""Render reviewed aggregate CSV tables to LaTeX, preserving source hashes."""
import argparse
import json
from pathlib import Path
import yaml
import pandas as pd
from opera.run.analyse_locked_predictions import file_hash, read_table


def tex(value):
    if pd.isna(value):
        return "Unavailable"
    chars = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
    return "".join(chars.get(c, c) for c in str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    prepared = []
    seen = set()
    skipped = []
    for spec in recipe["tables"]:
        name = spec["id"]
        if name in seen or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Table IDs must be unique safe filenames")
        seen.add(name)
        source = Path(spec["source"])
        if not source.is_absolute():
            source = args.config.resolve().parent / source
        if spec.get("optional") and not source.is_file():
            skipped.append(name)  # produced by a later step; listed, not fatal
            continue
        data = read_table(source)
        columns = spec["columns"]
        if not 1 <= len(columns) <= 6:
            raise ValueError("Readable printed tables require at most six columns; retain full CSV separately")
        if data.empty:
            raise ValueError(f"{name}: header-only template is not an empirical table")
        selected = data[columns]
        width = 0.95 / len(columns)
        fmt = "@{}" + (f"p{{{width:.3f}\\linewidth}}" * len(columns)) + "@{}"
        header = " & ".join(tex(spec.get("labels", {}).get(c, c)) for c in columns)
        lines = [r"\begingroup\small", r"\begin{longtable}{" + fmt + "}",
                 r"\caption{" + tex(spec["caption"]) + r"}\\",
                 r"\toprule " + header + r"\\\midrule\endfirsthead",
                 r"\toprule " + header + r"\\\midrule\endhead"]
        lines += [" & ".join(tex(v) for v in row) + r"\\" for row in selected.itertuples(index=False, name=None)]
        lines += [r"\bottomrule\end{longtable}", r"\endgroup"]
        prepared.append((name, "\n".join(lines), str(source), file_hash(source)))
    if not prepared:
        raise ValueError("Register real aggregate tables")
    if not args.execute:
        print(json.dumps({"tables": sorted(seen), "skipped": skipped, "preview": True}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    combined = []
    for name, content, source, sha in prepared:
        (args.output_dir / (name + ".tex")).write_text(content + "\n")
        combined.append(content)
        manifest.append({"id": name, "source": source, "source_sha": sha})
    (args.output_dir / "supplement_tables.tex").write_text("\n\n".join(combined) + "\n")
    (args.output_dir / "table_manifest.json").write_text(json.dumps({"recipe_sha": file_hash(args.config), "tables": manifest,
                    "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
