"""Our patients' records in the national (PHAIR) encoder's own format.

The national encoder was pretrained on the national MEDS extract with its own
code conventions, numeric normalisation and vocabulary. Rather than remapping
its embeddings onto our vocabulary, we take the same patients from that
national extract and tokenise them with the national vocabulary, so the
national and adapted national arms read records exactly as the encoder was
trained to. Outcomes, membership and splits stay those of the paper.

1. ``filter``: keep the national MEDS rows of the patients in our local MEDS,
   written under the local pretraining split of each patient (train/tuning), so
   local adaptation sees the same patients in the same split as local
   pretraining.
2. ``tokenize``: ``bonsai.run.create_data`` with the national vocabulary fixed
   (unknown codes become [UNK]) and the national data-creation settings.
3. ``audit``: patient coverage, [UNK] share and the latest event date, which
   must reach the test period so recent history is not missing.

    python -m opera.run.prepare_external_sequences filter --external-meds <national MEDS> \
        --local-meds <OPERA_PAPER_MEDS_DIR> --output-dir <root>/external_sequences
    python -m opera.run.prepare_external_sequences tokenize --external-vocab <vocabulary.pt> \
        --data-config <create_data config name> --output-dir <root>/external_sequences
    python -m opera.run.prepare_external_sequences audit --output-dir <root>/external_sequences \
        --membership <cohort membership>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SPLITS = ("train", "tuning")


def _data_root(root):
    root = Path(root)
    return root / "data" if (root / "data").is_dir() else root


def local_split_assignment(local_meds):
    """subject_id -> pretraining split, from the local MEDS shards."""
    import pyarrow.parquet as pq

    assignment = {}
    for split in SPLITS:
        for path in sorted((_data_root(local_meds) / split).rglob("*.parquet")):
            ids = pq.read_table(path, columns=["subject_id"]).column(0).unique().to_pylist()
            for subject in ids:
                if assignment.setdefault(subject, split) != split:
                    raise ValueError(f"Subject {subject} is in more than one local split")
    if not assignment:
        raise ValueError(f"No local MEDS shards under {local_meds}")
    return assignment


def filter_external_meds(external_meds, assignment, output_dir):
    """Write the national MEDS rows of our patients under their local split."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    output = Path(output_dir) / "meds" / "data"
    if output.exists():
        raise FileExistsError(f"{output} exists; use a fresh output directory")
    ids = {split: pa.array([s for s, v in assignment.items() if v == split])
           for split in SPLITS}
    shards = sorted(_data_root(external_meds).rglob("*.parquet"))
    if not shards:
        raise ValueError(f"No national MEDS shards under {external_meds}")
    found, rows, latest = set(), {split: 0 for split in SPLITS}, None
    for number, path in enumerate(shards):
        table = pq.read_table(path)
        for split in SPLITS:
            part = table.filter(pc.is_in(table["subject_id"], value_set=ids[split]))
            if not part.num_rows:
                continue
            target = output / split / f"part-{number:05d}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(part, target)
            rows[split] += part.num_rows
            found.update(part["subject_id"].unique().to_pylist())
            times = pc.max(part["time"]).as_py()
            latest = times if latest is None or (times and times > latest) else latest
        print(f"filtered {number + 1}/{len(shards)}: {path.name}", flush=True)
    report = {"local_patients": len(assignment), "found_in_national": len(found),
              "missing_from_national": len(assignment) - len(found),
              "rows_by_split": rows, "latest_event": str(latest)}
    (Path(output_dir) / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def tokenize_command(output_dir, external_vocab, data_config):
    output_dir = Path(output_dir).resolve()
    return [sys.executable, "-m", "bonsai.run.create_data",
            "--config-name", data_config,
            f"paths.input_dir={output_dir / 'meds'}",
            f"paths.output_dir={output_dir / 'sequences'}",
            f"tokenizer.vocabulary={Path(external_vocab).resolve()}",
            # The vocabulary is fixed, so no vocabulary-fitting cutoff applies.
            "vocabulary_cutoff_date=null",
            f"splits=[{','.join(SPLITS)}]"]


def audit_sequences(output_dir, membership):
    """Coverage of the paper population and [UNK] share in the national tokens."""
    import pandas as pd
    import torch

    sequences = Path(output_dir) / "sequences"
    vocabulary = torch.load(sequences / "vocabulary.pt", weights_only=False)
    unknown = vocabulary["[UNK]"]
    population = pd.read_parquet(membership) if str(membership).endswith(".parquet") \
        else pd.read_csv(membership)
    wanted = set(population.subject_id)
    covered, tokens, unknown_tokens = set(), 0, 0
    for split in SPLITS:
        for subject in torch.load(sequences / f"subject_data_{split}.pt", weights_only=False):
            if subject["subject_id"] in wanted:
                covered.add(subject["subject_id"])
                codes = torch.as_tensor(subject["code"])
                tokens += int(codes.numel())
                unknown_tokens += int((codes == unknown).sum())
    report = {"paper_patients": len(wanted), "with_national_sequence": len(covered),
              "missing": len(wanted - covered),
              "unknown_token_share": unknown_tokens / tokens if tokens else None,
              "vocabulary_size": len(vocabulary)}
    (Path(output_dir) / "audit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["missing"]:
        print("WARNING: some paper patients have no national sequence; they would be "
              "dropped from the national arms. Resolve before preparing stage 2.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("filter", "tokenize", "audit"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-meds", type=Path)
    parser.add_argument("--local-meds", type=Path)
    parser.add_argument("--external-vocab", type=Path)
    parser.add_argument("--data-config", default="daly_care_joined_bins_data",
                        help="create_data config with the national tokenisation settings")
    parser.add_argument("--membership", type=Path)
    args = parser.parse_args()
    if args.step == "filter":
        filter_external_meds(args.external_meds, local_split_assignment(args.local_meds),
                             args.output_dir)
    elif args.step == "tokenize":
        command = tokenize_command(args.output_dir, args.external_vocab, args.data_config)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
    else:
        audit_sequences(args.output_dir, args.membership)


if __name__ == "__main__":
    main()
