"""What the model's tokens mean: audit the vocabulary against the code dictionary.

Run on the server with the vocabulary of the locked model, before the
attribution summaries are read or labelled:

    python -m opera.run.audit_vocabulary \\
        --vocabulary $OPERA_SEQUENCE_DATA_DIR/vocabulary.pt \\
        --config opera/configs/attribution_analysis.yaml \\
        --attribution-summary <analysis root>/attributions/summary_365d \\
        --output-dir <analysis root>/vocabulary_audit

Writes (codes and names only; no patient data, so all files can be exported):

``code_meanings.csv``
    every token: source, code system, code, name, where the name came from
    (reviewed, table, built-in, group or code), clinical group and chapter.
``vocabulary_formats.csv``
    every token format (prefix and shape) with its source, number of tokens
    and three examples. Formats read as ``Other`` are not recognised and need a
    prefix in ``opera.interpretability.code_meaning.FAMILIES``.
``vocabulary_coverage.csv``
    per source, the share of tokens named at each step.
``concept_coverage.csv``
    every configured attribution finding and laboratory test with the tokens
    it matches, their names and bin range: review before setting
    ``concept_mapping_reviewed: true``.
``review_needed.csv``
    rolled-up concepts without a name, most important for the attributions
    first (with ``--attribution-summary``), for review or a code table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from opera.interpretability.attribution_summaries import rollup
from opera.interpretability.code_meaning import CodeDictionary


def load_vocabulary(path: Path) -> dict[str, int]:
    from opera.evaluation.vocabulary_embeddings import load_vocabulary as load

    return dict(load(path))


def token_format(token: str) -> str:
    """Prefix and shape: ``DIAGNOSIS//<A9..>``, ``LABTEST//<A9..>//BIN_<9>``."""
    parts = str(token).split("//")
    shape = [re.sub(r"\d+", "9", re.sub(r"[A-Za-z]+", "A", p)) for p in parts[1:]]
    shape = [s if s in {"A", "9"} else f"<{s[:6]}{'..' if len(s) > 6 else ''}>" for s in shape]
    tail = "//".join(shape)
    if len(parts) > 2 and re.match(r"^BIN_\d+$", parts[-1]):
        tail = "//".join(shape[:-1] + ["BIN_<9>"])
    return parts[0] + ("//" + tail if tail else "")


def audit(vocabulary: dict[str, int], dictionary: CodeDictionary,
          importance: pd.DataFrame | None = None):
    tokens = sorted(vocabulary, key=vocabulary.get)
    meanings = dictionary.table(tokens)
    meanings["format"] = meanings.token.map(token_format)
    formats = (meanings.groupby(["format", "source"], sort=False)
               .agg(tokens=("token", "size"), examples=("token", lambda t: "; ".join(t[:3])))
               .reset_index().sort_values("tokens", ascending=False))
    coverage = (pd.crosstab(meanings.source, meanings.name_source, normalize="index")
                .mul(100).round(1).assign(tokens=meanings.source.value_counts()))
    concepts = pd.DataFrame({"concept": sorted({rollup(t) for t in tokens
                                                if not t.startswith("[")})})
    concepts = concepts.join(dictionary.table(concepts.concept).drop(columns="token"))
    review = concepts.loc[concepts.name_source.isin(["code", "group"])
                          & ~concepts.source.isin(["Demographics"])]
    if importance is not None and len(importance):
        weight = (importance.assign(size=importance.mean_pp.abs()).groupby("concept")
                  .agg(outcomes=("outcome", "nunique"), largest_effect=("size", "max")))
        review = (review.merge(weight, left_on="concept", right_index=True, how="left")
                  .fillna({"outcomes": 0})
                  .sort_values(["outcomes", "largest_effect"], ascending=False))
    return meanings, formats, coverage.reset_index(), review


def concept_coverage(vocabulary, config, dictionary: CodeDictionary) -> pd.DataFrame:
    """For every configured finding and laboratory test: the tokens it matches.

    A finding that matches nothing, or a laboratory test without bins, would
    silently drop out of the attribution figures; review this table (and the
    names next to each code) before setting ``concept_mapping_reviewed``.
    """
    tokens = list(vocabulary)
    items = [{"kind": "finding", "name": c["name"], "label": c.get("label", c["name"]),
              "prefixes": list(c.get("codes", []))} for c in config.get("concepts") or []]
    items += [{"kind": "lab_test", "name": t["npu"], "label": t.get("label", t["npu"]),
               "prefixes": [f"LABTEST//{t['npu']}//BIN_"]} for t in config.get("lab_tests") or []]
    rows = []
    for item in items:
        hits = [t for t in tokens if any(t.startswith(p) for p in item["prefixes"])]
        bins = sorted(int(m.group(1)) for t in hits if (m := re.search(r"//BIN_(\d+)$", t)))
        base = sorted({rollup(t) for t in hits})
        rows.append({
            **{k: item[k] for k in ("kind", "name", "label")},
            "prefixes": "; ".join(item["prefixes"]), "tokens": len(hits),
            "bins": f"{bins[0]}-{bins[-1]}" if bins else "",
            "matched_codes": "; ".join(base[:8]) + (" ..." if len(base) > 8 else ""),
            "dictionary_names": "; ".join(dict.fromkeys(dictionary.name(b) for b in base[:8])),
            "status": "OK" if hits else "NO MATCH",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="Analysis config with concept_labels and code_tables")
    parser.add_argument("--attribution-summary", type=Path,
                        help="Summary directory with attribution_top_concepts.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search", help="Regular expression matched against every token and "
                        "its name (e.g. 'TP53|MYC|IGHV|17p'); hits go to search_hits.csv")
    args = parser.parse_args()
    if args.config and not args.config.is_file():
        parser.error(f"--config {args.config} is not a file (an empty variable expands to '.')")
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    dictionary = CodeDictionary.from_config(config)
    importance = None
    if args.attribution_summary:
        importance = pd.read_csv(args.attribution_summary / "attribution_top_concepts.csv")
    meanings, formats, coverage, review = audit(load_vocabulary(args.vocabulary), dictionary,
                                                importance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meanings.to_csv(args.output_dir / "code_meanings.csv", index=False)
    formats.to_csv(args.output_dir / "vocabulary_formats.csv", index=False)
    coverage.to_csv(args.output_dir / "vocabulary_coverage.csv", index=False)
    review.to_csv(args.output_dir / "review_needed.csv", index=False)
    coverage_of_concepts = concept_coverage(load_vocabulary(args.vocabulary), config, dictionary)
    coverage_of_concepts.to_csv(args.output_dir / "concept_coverage.csv", index=False)
    if args.search:
        hits = meanings.loc[meanings.token.str.contains(args.search, case=False, regex=True)
                            | meanings.name.str.contains(args.search, case=False, regex=True)]
        hits.to_csv(args.output_dir / "search_hits.csv", index=False)
        print(f"{len(hits)} tokens match {args.search!r} (search_hits.csv)")
        for row in hits.head(40).itertuples():
            print(f"  {row.token}  {row.name}")
    other = formats.loc[formats.source == "Other"]
    summary = {
        "tokens": len(meanings),
        "named_share": round(float(meanings.name_source.isin(
            ["reviewed", "table", "built-in"]).mean()), 3),
        "unrecognised_formats": other.format.tolist(),
        "demographic_formats": formats.loc[formats.source == "Demographics", "format"].tolist(),
        "code_tables": [spec.get("system") for spec in config.get("code_tables") or []],
        "concepts_to_review": len(review),
        "configured_findings_without_match": coverage_of_concepts.loc[
            coverage_of_concepts.status != "OK", "label"].tolist(),
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(other):
        print("Unrecognised token formats (read as 'Other'): add their prefixes to FAMILIES")


if __name__ == "__main__":
    main()
