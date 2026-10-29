"""What the model's tokens mean: audit the vocabulary against the code dictionary.

Run on the server with the vocabulary of the locked model, before the
attribution summaries are read or labelled:

    python -m opera.run.audit_vocabulary \\
        --vocabulary $OPERA_SEQUENCE_DATA_DIR/vocabulary.pt \\
        --config opera/configs/attribution_analysis.yaml \\
        --attribution-summary <analysis root>/attributions/summary_365d \\
        --output-dir <analysis root>/vocabulary_audit

Writes vocabulary/code audits without patient rows. Optional laboratory-bin
definitions contain fitted distribution summaries and need the usual export
review alongside the figures:

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
import math
from pathlib import Path

import pandas as pd
import yaml

from opera.interpretability.attribution_summaries import concept_code_mask, rollup
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
              "prefixes": list(c.get("codes", [])), "exact_codes": bool(c.get("exact_codes"))}
             for c in config.get("concepts") or []]
    items += [{"kind": "lab_test", "name": t["npu"], "label": t.get("label", t["npu"]),
               "prefixes": [f"LABTEST//{t['npu']}//BIN_"]} for t in config.get("lab_tests") or []]
    rows = []
    for item in items:
        codes = pd.Series(tokens, dtype=str)
        hits = codes.loc[concept_code_mask(codes, {"codes": item["prefixes"],
                          "exact_codes": item.get("exact_codes", False)})].tolist()
        bins = sorted(int(m.group(1)) for t in hits if (m := re.search(r"//BIN_(\d+)$", t)))
        base = sorted({rollup(t) for t in hits})
        rows.append({
            **{k: item[k] for k in ("kind", "name", "label")},
            "prefixes": "; ".join(item["prefixes"]), "tokens": len(hits),
            "match_mode": "exact" if item.get("exact_codes") else "prefix",
            "bins": f"{bins[0]}-{bins[-1]}" if bins else "",
            "matched_codes": "; ".join(base[:8]) + (" ..." if len(base) > 8 else ""),
            "dictionary_names": "; ".join(dict.fromkeys(dictionary.name(b) for b in base[:8])),
            "status": "OK" if hits else "NO MATCH",
        })
    return pd.DataFrame(rows)


def adaptive_concept_coverage(mapping, vocabulary, config):
    """Explain lost molecular specificity without relabelling mixed ancestors."""
    required = {'code', 'adaptive/mapped_code'}
    if not required <= set(mapping):
        raise ValueError(f'Adaptive mapping lacks {sorted(required - set(mapping))}')
    if mapping.code.duplicated().any():
        raise ValueError('Adaptive mapping contains duplicate original codes')
    mapped = mapping.set_index('code')['adaptive/mapped_code'].astype(str)
    originals = mapping.groupby('adaptive/mapped_code').code.agg(list).to_dict()
    rows = []
    for concept in config.get('concepts', []):
        if not concept.get('exact_codes'):
            continue
        expected = set(concept['codes'])
        for code in concept['codes']:
            target = mapped.get(code)
            other = [] if target is None else [c for c in originals.get(target, []) if c not in expected]
            status = ('absent_from_mapping' if target is None else
                      'mapped_token_missing' if target not in vocabulary else
                      'retained_exact' if target == code else
                      'collapsed_mixed_codes' if other else 'collapsed_within_concept_only')
            rows.append(dict(concept=concept['name'], source_code=code, mapped_code=target,
                             status=status, other_original_codes=' | '.join(sorted(other)),
                             action='Keep exact mapping' if status == 'retained_exact' else
                             'Review mapping; do not infer molecular status from a shared prefix'))
    return pd.DataFrame(rows)


def laboratory_bin_definitions(metadata, vocabulary):
    """Physical boundaries implied by the saved ehr2meds transform, in input units.

    The 1st/99th percentile normalization clips tails. End bins consequently
    extend to infinity; their finite boundary is not a clinical reference limit.
    """
    needed = {'code', 'numeric/bin_1', 'numeric/bin_99', 'numeric/bin_edges'}
    if not needed <= set(metadata):
        raise ValueError(f'Numeric metadata lacks {sorted(needed - set(metadata))}')
    rows = []
    for item in metadata.to_dict('records'):
        code = item['code']
        if not str(code).startswith('LABTEST//'):
            continue
        low, high, edges = item['numeric/bin_1'], item['numeric/bin_99'], item['numeric/bin_edges']
        if pd.isna(low) or pd.isna(high) or edges is None or not hasattr(edges, '__iter__'):
            continue
        edges = list(edges)
        if not all(0 <= e <= 1 for e in edges) or edges != sorted(set(edges)):
            raise ValueError(f'Invalid normalized bin edges for {code}')
        low, high = float(low), float(high)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f'Non-finite normalization bounds for {code}')
        physical = [low + float(e) * (high-low) for e in edges]
        boundaries = [float('-inf'), *physical, float('inf')]
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            # Clipping happens before right-sided bin assignment. An edge at
            # zero leaves bin 0 unreachable, rather than capturing the low tail.
            # A constant assay always normalizes to zero, whatever its raw value.
            if high <= low:
                reachable = index == sum(edge <= 0 for edge in edges)
                left, right = ((float('-inf'), float('inf')) if reachable
                               else (float('nan'), float('nan')))
            else:
                normalized_left = max(0., edges[index-1] if index else 0.)
                normalized_right = min(1., edges[index] if index < len(edges) else 1.)
                reachable = normalized_left < normalized_right or (
                    index == len(edges) and normalized_left == 1.)
                if not reachable:
                    left = right = float('nan')
                elif normalized_left == 0:
                    left = float('-inf')
            token = f'{code}//BIN_{index}'
            rows.append(dict(code=code, token=token, bin_index=index, fitted_bin_count=len(edges)+1,
                             lower_inclusive=left, upper_exclusive=right,
                             reachable_after_clipping=reachable,
                             token_in_vocabulary=token in vocabulary,
                             normalization_lower=low, normalization_upper=high,
                             unit_status='Input values as encoded; verify source units',
                             interpretation='Quantile bin, not a clinical reference interval'))
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
    parser.add_argument('--adaptive-mapping', type=Path,
                        help='Frozen ehr2meds adaptive_code_mapping.parquet (or CSV)')
    parser.add_argument('--numeric-metadata', type=Path,
                        help='Frozen ehr2meds numeric_metadata.json (or Parquet)')
    args = parser.parse_args()
    if args.config and not args.config.is_file():
        parser.error(f"--config {args.config} is not a file (an empty variable expands to '.')")
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    dictionary = CodeDictionary.from_config(config)
    importance = None
    if args.attribution_summary:
        importance = pd.read_csv(args.attribution_summary / "attribution_top_concepts.csv")
    vocabulary = load_vocabulary(args.vocabulary)
    meanings, formats, coverage, review = audit(vocabulary, dictionary,
                                                importance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meanings.to_csv(args.output_dir / "code_meanings.csv", index=False)
    formats.to_csv(args.output_dir / "vocabulary_formats.csv", index=False)
    coverage.to_csv(args.output_dir / "vocabulary_coverage.csv", index=False)
    review.to_csv(args.output_dir / "review_needed.csv", index=False)
    coverage_of_concepts = concept_coverage(load_vocabulary(args.vocabulary), config, dictionary)
    coverage_of_concepts.to_csv(args.output_dir / "concept_coverage.csv", index=False)
    if args.adaptive_mapping:
        mapping = (pd.read_parquet(args.adaptive_mapping) if args.adaptive_mapping.suffix == '.parquet'
                   else pd.read_csv(args.adaptive_mapping))
        adaptive_concept_coverage(mapping, vocabulary, config).to_csv(
            args.output_dir / 'adaptive_concept_coverage.csv', index=False)
    if args.numeric_metadata:
        metadata = (pd.read_parquet(args.numeric_metadata) if args.numeric_metadata.suffix == '.parquet'
                    else pd.read_json(args.numeric_metadata))
        laboratory_bin_definitions(metadata, vocabulary).to_csv(
            args.output_dir / 'laboratory_bin_definitions.csv', index=False)
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
