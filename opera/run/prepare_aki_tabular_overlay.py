"""Link unchanged tabular fits and corrected AKI fits into one analysis root.

Links refer to original fit directories, including their untouched provenance.
Missing corrected fits stay missing; there is no fallback to the old AKI fits.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from opera.run.audit_aki_correction import AKI, HORIZONS
from opera.run.decide_outcome_panel import sha256_file
from opera.run.paper_tabular_queue import pooled_tabular_jobs, tabular_jobs


def prepare(bundle: Path, old_root: Path, new_root: Path, output: Path) -> dict:
    old_root, new_root, output = old_root.resolve(), new_root.resolve(), output.resolve()
    if output.exists() or old_root == new_root or output in (old_root, new_root):
        raise ValueError("Require distinct old/new tabular roots and a fresh overlay")
    amended = json.loads((bundle / "bundle.json").read_text())
    if amended.get("outcome_amendment", {}).get("changed_outcomes") != [AKI]:
        raise ValueError("Require the combined AKI repair bundle")
    jobs = [*tabular_jobs(bundle, HORIZONS), *pooled_tabular_jobs(bundle, HORIZONS)]
    if not jobs:
        raise ValueError("No specialist-derived tabular jobs in bundle")
    links = []
    for job in jobs:
        root = new_root if job["outcome"] == AKI else old_root
        destination = root / job["relative_dir"]
        marker = destination / "training_complete.json"
        if marker.exists() and json.loads(marker.read_text())["inputs"]["job"] != job:
            raise ValueError(f"Tabular fit belongs to different inputs: {destination}")
        links.append({"relative_dir": job["relative_dir"], "source": str(destination),
                      "action": "corrected_AKI" if job["outcome"] == AKI else "carry_forward",
                      "complete_at_preparation": marker.is_file()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aki_tabular_", dir=output.parent) as temporary:
        staged = Path(temporary) / "overlay"
        staged.mkdir()
        for link in links:
            path = staged / link["relative_dir"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(link["source"], target_is_directory=True)
        report = {"bundle_manifest_sha256": sha256_file(bundle / "bundle.json"),
                  "old_root": str(old_root), "corrected_root": str(new_root),
                  "no_old_AKI_fallback": True, "links": links}
        (staged / "carry_forward_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(output)
    return {"overlay": str(output), "linked_jobs": len(links), "no_old_AKI_fallback": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--old-tabular-root", type=Path, required=True)
    parser.add_argument("--corrected-tabular-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle_dir, args.old_tabular_root,
                             args.corrected_tabular_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
