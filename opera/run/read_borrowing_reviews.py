"""Read filled-in borrowing-pair review sheets (fillable PDFs) into tables.

Each sheet (paper/review/borrowing_pairs_review.pdf, generated from
opera/configs/borrowing_hypotheses.yaml) has, per pair, a choice field
``view<ID>`` (agree / disagree / unsure) and a text field ``comment<ID>``, where
``<ID>`` is H1, H2, ... for the expected pairs and C1, ... for the negative
controls, in the order of the YAML; and free-text fields for the reviewer's
name, missed pairs and other comments. Writes one row per reviewer and pair,
one row per reviewer for the free text, and prints a tally per pair.

    python -m opera.run.read_borrowing_reviews replies/*.pdf --output-dir replies/
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
VIEWS = ("agree", "disagree", "unsure")


def _field(fields: dict, *names: str) -> str:
    """Value of the first field present under any spelling (LaTeX drops '_')."""
    for name in names:
        for key in (name, name.replace("_", "")):
            if key in fields:
                value = fields[key].get("/V") if isinstance(fields[key], dict) else fields[key]
                return "" if value is None else str(value).strip()
    return ""


def items(plan: dict) -> list[dict]:
    return ([{"item": f"H{i}", "kind": "expected pair", **e}
             for i, e in enumerate(plan.get("hypotheses", []), 1)]
            + [{"item": f"C{i}", "kind": "negative control", **e}
               for i, e in enumerate(plan.get("negative_controls", []), 1)])


def read_sheet(path: Path, plan: dict) -> tuple[list[dict], dict]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise SystemExit("Reading the review PDFs needs pypdf: pip install pypdf") from exc

    fields = PdfReader(str(path)).get_fields() or {}
    reviewer = _field(fields, "reviewer") or path.stem
    rows = []
    for entry in items(plan):
        view = _field(fields, f"view_{entry['item']}").lower()
        rows.append({
            "reviewer": reviewer, "file": path.name, "item": entry["item"],
            "kind": entry["kind"], "source": entry["source"], "target": entry["target"],
            "domain": entry["domain"],
            "view": view if view in VIEWS else ("" if not view else f"unrecognised: {view}"),
            "comment": _field(fields, f"comment_{entry['item']}"),
        })
    free = {"reviewer": reviewer, "file": path.name,
            "missed_pairs": _field(fields, "missed_pairs"),
            "other_comments": _field(fields, "other_comments")}
    return rows, free


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", type=Path, nargs="+", help="Filled-in review PDFs")
    parser.add_argument("--plan", type=Path,
                        default=REPO / "opera/configs/borrowing_hypotheses.yaml",
                        help="The YAML the sheets were generated from")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    answers, free = [], []
    for sheet in args.sheets:
        rows, text = read_sheet(sheet, plan)
        answers += rows
        free.append(text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("review_answers.csv", answers), ("review_free_text.csv", free)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(f"{len(args.sheets)} sheets; per pair (agree / disagree / unsure / blank):")
    for entry in items(plan):
        tally = Counter(r["view"] for r in answers if r["item"] == entry["item"])
        print(f"  {entry['item']:>3} {entry['source']} -> {entry['target']}, {entry['domain']}: "
              f"{tally['agree']} / {tally['disagree']} / {tally['unsure']} / {tally['']}")
    print(f"Written: {args.output_dir / 'review_answers.csv'}, "
          f"{args.output_dir / 'review_free_text.csv'}")


if __name__ == "__main__":
    main()
