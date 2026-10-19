"""Code -> name table from DALY-CARE's own source data.

Most DALY-CARE source tables carry the text next to the code: laboratory
results with the NPU code and the analysis name, prescriptions and
administrations with the ATC code and the drug name, pathology with the SNOMED
code and its text. This reads only those two columns, keeps for every code its
most frequent name, and writes ``code,label`` for ``code_tables`` in the
attribution config (opera/resources/code_systems/README.md). Codes seen in
fewer than ``--min-count`` rows are left out, so free text typed for a single
record cannot reach the table; the output holds classification text only.

    python -m opera.run.build_code_table --system npu \\
        --input /data/lab.parquet --code-column analysiscode \\
        --label-column analysisname --output resources/npu_daly_care.csv

Then, in the attribution config:

    code_tables:
      - {system: npu, path: resources/npu_daly_care.csv, code: code, label: label}

Several inputs for one system are pooled (e.g. prescriptions and hospital
administrations for ATC). Excel and delimited files work as well as parquet.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from opera.interpretability.code_meaning import normalise

SYSTEMS = ("icd10", "atc", "npu", "sks_procedure", "snomed")


def _batches(path: Path, columns, *, sep=None, sheet=0, chunk_rows=1_000_000):
    """The two columns of a file, in chunks (lab tables have tens of millions of rows)."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows, columns=columns):
            yield batch.to_pandas()
    elif suffix in {".xlsx", ".xls"}:
        yield pd.read_excel(path, dtype=str, sheet_name=sheet, usecols=columns)
    else:
        yield from pd.read_csv(path, dtype=str, sep=sep, engine="python", usecols=columns,
                               chunksize=chunk_rows)


def count_pairs(paths, code_column, label_column, **reader):
    """Rows per (code, name) across all inputs."""
    counts = Counter()
    for path in paths:
        for frame in _batches(Path(path), [code_column, label_column], **reader):
            frame = frame.dropna()
            frame = frame.loc[(frame[code_column].astype(str).str.strip() != "")
                              & (frame[label_column].astype(str).str.strip() != "")]
            pairs = frame.groupby([code_column, label_column], sort=False).size()
            for (code, label), n in pairs.items():
                counts[(str(code).strip(), " ".join(str(label).split()))] += int(n)
    return counts


def build_table(counts, system, *, min_count=5):
    """One row per normalised code: its most frequent name and how many names it had."""
    by_code = {}
    for (code, label), n in counts.items():
        by_code.setdefault(normalise(system, code), Counter())[label] += n
    rows = []
    for code, labels in sorted(by_code.items()):
        if sum(labels.values()) < min_count:
            continue
        label, _ = max(labels.items(), key=lambda item: (item[1], item[0]))
        rows.append({"code": code, "label": label, "names_seen": len(labels)})
    return pd.DataFrame(rows, columns=["code", "label", "names_seen"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--code-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--min-count", type=int, default=5,
                        help="Leave out codes seen in fewer rows (keeps one-off free text out)")
    parser.add_argument("--sep", help="Delimiter for text files (default: sniffed)")
    parser.add_argument("--sheet", default=0, help="Excel sheet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = count_pairs(args.input, args.code_column, args.label_column, sep=args.sep,
                         sheet=args.sheet)
    table = build_table(counts, args.system, min_count=args.min_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} {args.system} codes named -> {args.output}")
    several = table.loc[table.names_seen > 1]
    if len(several):
        print(f"{len(several)} codes had more than one name; the most frequent is kept "
              "(names_seen column)")


if __name__ == "__main__":
    main()
