"""Inspect raw MEDS code namespaces before defining Table 1 prefix mappings."""

import argparse
import itertools
import json
from pathlib import Path

from opera.functional.table1_meds import raw_meds_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meds-dir", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=3)
    args = parser.parse_args()
    if args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    root = args.meds_dir
    candidates = [
        root / "metadata" / "codes.parquet",
        root.parent / "metadata" / "codes.parquet",
    ]
    codebook = next((p for p in candidates if p.exists()), None)
    if codebook:
        import pandas as pd

        codes = pd.read_parquet(codebook, columns=["code"])["code"].dropna().astype(str)
        source = str(codebook.resolve())
    else:
        codes = set()
        for batch in itertools.islice(raw_meds_batches(root), args.max_batches):
            codes.update(batch.code.dropna().astype(str))
        source = (
            f"sample of at most {args.max_batches} raw MEDS batches (not exhaustive)"
        )
    groups = {}
    for code in sorted(set(codes)):
        namespace = code.split("//", 1)[0] if "//" in code else code.split("/", 1)[0]
        groups.setdefault(namespace, [])
        if len(groups[namespace]) < 5:
            groups[namespace].append(code)
    print(
        json.dumps(
            {
                "meds_dir": str(root.resolve()),
                "source": source,
                "unique_codes_inspected": len(set(codes)),
                "namespace_examples": groups,
            },
            indent=2,
        )
    )
    print(
        "Code vocabulary only; no patient IDs. Verify clinical namespaces before selecting prefixes."
    )


if __name__ == "__main__":
    main()
