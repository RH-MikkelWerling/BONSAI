"""Preflight for a new ehr2meds MEDS release, before ``create_data``.

Checks everything that would otherwise fail late or, worse, silently:

- layout: ``data/train`` and ``data/tuning`` shards, no ``data/held_out``
  (the physical split is 90/10; study splits come from the index date);
- ehr2meds sidecars: numeric metadata and the adaptive code mapping summary;
- numeric representation: laboratory results joined with their bin
  (``LABTEST//<NPU>//BIN_<k>``, 0-based) or their text value, the form the
  paper's encoders read (``configs/daly_care_joined_bins_data.yaml``);
- token formats: every code prefix is a known source
  (:mod:`opera.interpretability.code_meaning`), demographics present;
- shard order: subject, then time, as BONSAI preserves it;
- identities: cohort membership and every outcome file refer to subjects in
  the release, and (``--old-mapping``/``--new-mapping``) the integer subject
  IDs did not change between ehr2meds runs: if they did, existing outcome and
  membership files point at other patients;
- vocabulary: share of the release's codes known to the current vocabulary.

Prints counts and verdicts only (no identifiers), and writes the same as JSON.
Exit status 1 when any check fails.

    python -m opera.run.check_meds_release --meds-dir $EHR2MEDS_OUTPUT \\
        --membership $BONSAI_COHORT_MEMBERSHIP --outcomes-dir $BONSAI_OUTCOMES_DIR \\
        --old-vocab $OPERA_PAPER_LOCAL_VOCAB --output meds_release_check.json

``--old-mapping``/``--new-mapping`` are needed only when ehr2meds renumbers
subjects (``subject_id_mapping`` in the preMEDS config); the production config
keeps the pseudonymised identifiers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from opera.interpretability.code_meaning import family

BIN = re.compile(r"//BIN_(\d+)$")
SIDECARS = ("numeric_metadata.json", "adaptive_code_mapping.summary.json")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, status, detail, **numbers):
        self.rows.append({"check": check, "status": status, "detail": detail, **numbers})

    @property
    def failed(self):
        return any(r["status"] == "FAIL" for r in self.rows)


def shards(root: Path, split: str) -> list[Path]:
    return sorted((root / "data" / split).glob("*.parquet"))


def check_layout(root: Path, report: Report):
    print(f"MEDS release: {root.resolve()}")
    for split in ("train", "tuning"):
        files = shards(root, split)
        hint = ""
        if not files and list((root / split).glob("*.parquet")):
            hint = f"; found {split}/*.parquet at the root: an older flat release, not ehr2meds' data/ layout"
        elif not files:
            hint = " (is --meds-dir / EHR2MEDS_OUTPUT the new release?)"
        report.add(f"layout:{split}", "PASS" if files else "FAIL",
                   f"{len(files)} shards in data/{split}{hint}", shards=len(files))
    held = root / "data" / "held_out"
    report.add("layout:no_held_out", "FAIL" if held.exists() else "PASS",
               "data/held_out exists: the default 80/10/10 split was used; rerun with "
               "train 0.9 / tuning 0.1" if held.exists() else "no physical held-out split")
    found = {name: [str(p.relative_to(root)) for p in root.rglob(name)] for name in SIDECARS}
    for name, paths in found.items():
        report.add(f"sidecar:{name}", "PASS" if paths else "WARN",
                   ", ".join(paths) if paths else "not found (archive it with the release)")


def scan(root: Path):
    files = shards(root, "train") + shards(root, "tuning")
    frame = pl.scan_parquet([str(f) for f in files])
    columns = frame.collect_schema().names()
    codes = (frame.group_by("code").agg(pl.len().alias("n"),
                                        pl.col("time").is_null().sum().alias("untimed"))
             .collect())
    subjects = frame.select(pl.col("subject_id").unique()).collect()["subject_id"]
    numeric = [c for c in ("numeric_value", "numeric_value_bin", "numeric_value_normalized")
               if c in columns]
    # Numeric laboratory events left without a bin suffix, split by why:
    # a bin exists but was not joined (a pipeline error), or no bin was fitted
    # for the test (e.g. first seen after the cutoff: harmless, read as the
    # bare test code). Categorical results carry no numeric value.
    bare = {"unjoined": 0, "no_bin": 0, "lab_events": 0}
    if "numeric_value" in columns:
        unsuffixed = (pl.col("code").str.starts_with("LABTEST//")
                      & ~pl.col("code").str.contains(r"//BIN_\d+$")
                      & pl.col("numeric_value").is_not_null())
        has_bin = (pl.col("numeric_value_bin").is_not_null() if "numeric_value_bin" in columns
                   else pl.lit(False))
        counts = frame.select(
            (unsuffixed & has_bin).sum().alias("unjoined"),
            (unsuffixed & ~has_bin).sum().alias("no_bin"),
            pl.col("code").str.starts_with("LABTEST//").sum().alias("lab_events"),
        ).collect().row(0, named=True)
        bare = {k: int(v) for k, v in counts.items()}
    return files, columns, codes.to_pandas(), set(subjects.to_list()), numeric, bare


def check_representation(columns, codes: pd.DataFrame, numeric, bare, report: Report):
    report.add("schema", "PASS" if {"subject_id", "time", "code"} <= set(columns) else "FAIL",
               f"columns: {', '.join(columns)}")
    lab = codes.loc[codes.code.str.startswith("LABTEST//")]
    binned = lab.code.str.contains(r"//BIN_\d+$")
    parts = lab.code.str.count("//")
    text = ~binned & (parts >= 2)
    events = max(int(lab.n.sum()), 1)
    share = float(lab.loc[binned, "n"].sum()) / events
    report.add("labs:joined_bins", "PASS" if binned.any() else "FAIL",
               f"{share:.1%} of laboratory events carry //BIN_k, "
               f"{float(lab.loc[text, 'n'].sum()) / events:.1%} a text value",
               binned_codes=int(binned.sum()), text_codes=int(text.sum()))
    report.add("labs:bins_joined", "PASS" if bare["unjoined"] == 0 else "FAIL",
               f"{bare['unjoined']} laboratory events have a bin that was not joined to "
               "the code (join_numeric_bins missing)", events=bare["unjoined"])
    share = bare["no_bin"] / max(bare["lab_events"], 1)
    report.add("labs:numeric_without_bin", "PASS" if share < 0.01 else "WARN",
               f"{bare['no_bin']} numeric laboratory events ({share:.3%}) have no fitted "
               "bin and are read as the bare test code", events=bare["no_bin"])
    bins = lab.loc[binned, "code"].str.extract(BIN)[0].astype(int)
    report.add("labs:bins_zero_based", "PASS" if len(bins) and bins.min() == 0 else "WARN",
               f"bin indices {bins.min() if len(bins) else '-'} to "
               f"{bins.max() if len(bins) else '-'}")
    lower = codes.code.str.contains(r"//bin_\d+$").sum()
    report.add("labs:suffix_case", "PASS" if lower == 0 else "FAIL",
               f"{lower} codes use lower-case //bin_k; the attribution and vocabulary "
               "code expects //BIN_k")
    if numeric:
        report.add("numeric_columns", "PASS",
                   f"{', '.join(numeric)} present; joined_binning mode drops them, so "
                   "the encoder reads the joined code only")


def check_formats(codes: pd.DataFrame, report: Report):
    sources = codes.assign(source=codes.code.map(lambda c: family(c)[0]))
    by_source = sources.groupby("source").n.sum()
    total = max(int(by_source.sum()), 1)
    other = sources.loc[sources.source == "Other"]
    formats = sorted({str(c).split("//")[0] for c in other.code})
    report.add("formats:known_sources",
               "PASS" if other.n.sum() / total < 0.001 else "WARN",
               "events by source: " + ", ".join(f"{k} {v / total:.1%}"
                                                 for k, v in by_source.items())
               + (f"; unrecognised prefixes: {', '.join(formats[:15])}" if formats else ""),
               unrecognised_codes=len(other))
    demo = sources.loc[sources.source == "Demographics"]
    report.add("formats:demographics", "PASS" if len(demo) else "FAIL",
               f"{len(demo)} demographic codes ({', '.join(sorted({str(c).split('//')[0] for c in demo.code}))}), "
               f"{int(demo.untimed.sum())} untimed events")


def check_order(files, report: Report, limit=8):
    bad = 0
    for path in files[:limit]:
        frame = pl.read_parquet(path, columns=["subject_id", "time"])
        ordered = frame.sort(["subject_id", "time"], nulls_last=False, maintain_order=True)
        bad += int(not frame.equals(ordered))
    report.add("order", "PASS" if bad == 0 else "FAIL",
               f"{bad} of {min(limit, len(files))} checked shards not ordered by subject, time")


def _ids(path: Path, column="subject_id"):
    frame = pd.read_parquet(path, columns=[column]) if path.suffix == ".parquet" \
        else pd.read_csv(path, usecols=[column])
    return set(frame[column].dropna().astype("int64"))


def check_identities(subjects, membership, outcomes, report: Report):
    if membership:
        members = _ids(membership)
        covered = len(members & subjects) / max(len(members), 1)
        report.add("ids:membership_in_release", "PASS" if covered >= 0.99 else "FAIL",
                   f"{covered:.2%} of {len(members)} cohort patients have events in the "
                   "release", members=len(members))
    if outcomes:
        worst, missing = 1.0, []
        files = sorted(Path(outcomes).glob("*.parquet"))
        for path in files:
            ids = _ids(path)
            share = len(ids & subjects) / max(len(ids), 1)
            worst = min(worst, share)
            if share < 0.99:
                missing.append(path.stem)
        report.add("ids:outcomes_in_release", "PASS" if not missing else "FAIL",
                   f"{len(files)} outcome files; lowest share of their patients in the "
                   f"release {worst:.2%}" + (f"; below 99%: {', '.join(missing[:10])}"
                                             if missing else ""))


def check_mapping(old: Path, new: Path, report: Report):
    """Same personal identifier -> same integer subject_id in both runs."""
    a, b = pd.read_csv(old), pd.read_csv(new)
    key = next((c for c in a.columns if c in b.columns and c != "mapping"), None)
    value = "mapping" if "mapping" in a and "mapping" in b else None
    if key is None or value is None:
        report.add("ids:mapping_stable", "FAIL", "could not align the mapping files "
                   f"(columns {list(a.columns)} and {list(b.columns)})")
        return
    joined = a.merge(b, on=key, suffixes=("_old", "_new"))
    changed = int((joined[f"{value}_old"] != joined[f"{value}_new"]).sum())
    report.add("ids:mapping_stable", "PASS" if changed == 0 else "FAIL",
               f"{changed} of {len(joined)} shared persons changed subject_id "
               f"({len(a) - len(joined)} only in the old run, {len(b) - len(joined)} only "
               "in the new)" + ("; remap or rebuild membership and outcomes before use"
                                if changed else ""), changed=changed)


def check_vocabulary(codes: pd.DataFrame, old_vocab: Path, report: Report):
    from opera.evaluation.vocabulary_embeddings import load_vocabulary

    vocab = set(load_vocabulary(old_vocab))
    known = codes.code.isin(vocab)
    events = max(int(codes.n.sum()), 1)
    report.add("vocabulary:overlap", "PASS" if known.mean() > 0.5 else "WARN",
               f"{known.mean():.1%} of the release's codes ({codes.loc[known, 'n'].sum() / events:.1%} "
               "of events) are in the current vocabulary; a new vocabulary means the "
               "encoders are pretrained again", codes=len(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meds-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--outcomes-dir", type=Path)
    parser.add_argument("--old-vocab", type=Path)
    parser.add_argument("--old-mapping", type=Path)
    parser.add_argument("--new-mapping", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = Report()
    check_layout(args.meds_dir, report)
    if not any(r["check"].startswith("layout:t") and r["status"] == "FAIL" for r in report.rows):
        files, columns, codes, subjects, numeric, bare = scan(args.meds_dir)
        check_representation(columns, codes, numeric, bare, report)
        check_formats(codes, report)
        check_order(files, report)
        check_identities(subjects, args.membership, args.outcomes_dir, report)
        if args.old_vocab:
            check_vocabulary(codes, args.old_vocab, report)
        report.add("release:size", "PASS", f"{len(subjects)} subjects, {len(codes)} distinct "
                   f"codes, {int(codes.n.sum())} events", subjects=len(subjects))
    if args.old_mapping and args.new_mapping:
        check_mapping(args.old_mapping, args.new_mapping, report)
    for row in report.rows:
        print(f"{row['status']:4s}  {row['check']:28s} {row['detail']}")
    if args.output:
        args.output.write_text(json.dumps(report.rows, indent=2, default=str))
    print("FAILED" if report.failed else "Ready for create_data")
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
