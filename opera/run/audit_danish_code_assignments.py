"""Reproduce the public-code audit without reading patient data or vocabulary.

Requires downloaded SKScomplete.txt, npu-codes-latest.csv and Patobank HTML
codebooks; see docs/DANISH_CODE_AUDIT_20261028.md for sources and invocation.
Code existence is checked mechanically. Clinical equivalence, laboratory bins
and coverage in the fitted model are separate questions, not inferred here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import yaml

from opera.interpretability.code_meaning import family, normalise

SKS_URL = "https://filer.sundhedsdata.dk/sks/data/skscomplete/SKScomplete.txt"
NPU_URL = "https://cms.ifcc.org/wp-content/uploads/npu-codes-latest.csv"
SOURCES = {
    "SKScomplete.txt": SKS_URL,
    "npu-codes-latest.csv": NPU_URL,
    "patobank_molecular.html": "https://www.patobank.dk/kodebog/?tag=V",
    "patobank_blood.html": "https://www.patobank.dk/kodebog/?tag=D",
    "patobank_lymphoid.html": "https://www.patobank.dk/kodebog/?tag=E",
}
NOTES = {
    "BOQA": "Blood transfusion, not product-specific proof of red-cell transfusion.",
    "BOQA0": "Volume-based blood-transfusion code, not product-specific proof of red cells.",
    "BGDA2": "High-flow oxygen; valid from 2020-01-01. Not invasive mechanical ventilation.",
    "I48": "Includes flutter and fibrillation. Endpoint retains the old atrial_fibrillation identifier.",
    "I80": "Includes superficial phlebitis/thrombophlebitis; full category is not specific to deep VTE.",
    "G45": "Includes related transient cerebral syndromes, not exclusively classic TIA.",
    "C88": "Malignant immunoproliferative diseases; now a separate display group.",
    "B44": "Includes colonisation (DB448A) and non-invasive forms. Category is not specific to invasive disease.",
    "B59": "Danish code closed 2018-12-31. B485 (Pneumocystose) valid from 2019-01-01; R endpoint still uses B59 alone.",
    "B485": "Danish Pneumocystose code valid from 2019-01-01. Display grouping now includes B59 and B485.",
    "D47": "Includes uncertain/unknown-behaviour conditions. Not synonymous with a new malignancy.",
    "H02": "Official umbrella also contains H02C antiadrenal preparations.",
    "J01D": "Other beta-lactam antibacterials, broader than cephalosporins and carbapenems.",
    "L01X": "Historical ATC reclassification affects coverage; do not equate all targeted drugs with L01X.",
    "FE16E3": "Mutation recorded; code alone does not give variant pathogenicity or allele fraction.",
    "FE16E4": "Deletion recorded; may duplicate FK17D0 in the same patient.",
    "FK17D0": "Positive del(17p13) TP53 finding; do not confuse with negative FK17D1.",
    "FE13N8": "IGHV not hypermutated. FE13N0 (IGH mutation not detected) is not an equivalent.",
    "FE18UA": "MYC translocation detected, not MYC mutation/amplification or protein expression.",
    "J08141": "t(8;14) detected; code label does not specify genes/breakpoints. Candidate only; excluded pending local interpretation.",
    "FE16E2": "Insertion detected. Excluded from the narrow mutation/deletion concept pending review.",
    "FE16E5": "Amplification detected. Not included in the mutation/deletion concept.",
    "FE16E9": "Variant of unknown clinical significance. Not included as pathogenic TP53 abnormality.",
    "NPU19123": "Not found in this public NPU release. Previous draft LDH assignment is not retained.",
    "KTSF": "Former synthetic prototype biopsy label had no matching SKS code/prefix. Replaced by verified KTNE25A.",
}


def read_sks(path):
    """All historical records, using SDS's fixed-width specification, in cp1252."""
    rows = []
    for line in Path(path).read_text(encoding="cp1252").splitlines():
        if len(line) < 167:
            raise ValueError("Truncated SKS record; expected fixed-width data")
        rows.append({"code": line[3:23].strip(), "valid_from": line[23:31],
                     "change_date": line[31:39], "valid_to": line[39:47],
                     "label": line[47:167].strip()})
    return pd.DataFrame(rows)


class PatobankTable(HTMLParser):
    """Only adjacent table cells, never narrative matches or guessed code names."""
    def __init__(self):
        super().__init__()
        self.rows, self.cells, self.text = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        elif tag == "td":
            self.text = []

    def handle_data(self, data):
        if self.text is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.text is not None:
            self.cells.append(" ".join("".join(self.text).split()))
            self.text = None
        elif tag == "tr" and len(self.cells) >= 2:
            code, label = self.cells[:2]
            if re.fullmatch(r"[A-ZÆ][A-Z0-9]{5}", code):
                self.rows.append((code, label))


def inventory(root):
    rows = []

    def add(path, context, label, system, code, mode="prefix"):
        rows.append({"file": str(path), "assignment": context, "assigned_label": label,
                     "system": system, "code": normalise(system, code), "match_mode": mode})

    config_path = Path("opera/configs/attribution_analysis.example.yaml")
    config = yaml.safe_load((root / config_path).read_text())
    for c in config["concepts"]:
        for token in c["codes"]:
            _, system, rest = family(token)
            add(config_path, f"concept:{c['name']}", c["label"], system,
                rest.split("//")[0], "exact" if c.get("exact_codes") else "prefix")
    for token, label in config["summary"]["concept_labels"].items():
        _, system, rest = family(token)
        add(config_path, "summary.concept_labels", label, system, rest.split("//")[0])
    for test in config["lab_tests"]:
        add(config_path, "lab_tests", test["label"], "npu", test["npu"], "exact")

    group_path = Path("opera/resources/code_systems/clinical_groups.yaml")
    groups = yaml.safe_load((root / group_path).read_text())
    for kind, system in [("diagnoses", "icd10"), ("medication", "atc"),
                         ("procedures", "sks_procedure")]:
        for label, codes in groups[kind].items():
            for code in codes:
                if "-" in code:
                    start, end = code.split("-")
                    for n in range(int(start[1:]), int(end[1:]) + 1):
                        add(group_path, f"{kind}:{code}", label, system, f"{start[0]}{n:02}")
                else:
                    add(group_path, kind, label, system, code)

    atc_path = Path("opera/resources/code_systems/atc_groups.csv")
    for row in pd.read_csv(root / atc_path).itertuples():
        add(atc_path, "ATC anatomical/therapeutic group", row.name, "atc", row.code)

    prototype = Path("opera/run/render_paper_figure_prototypes.py")
    tree = ast.parse((root / prototype).read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in {"FACTOR_CODES", "CONCEPT_LABELS"}:
            continue
        values = ast.literal_eval(node.value)
        pairs = values.items() if name == "CONCEPT_LABELS" else (
            (token, f"Synthetic factor example: {factor}; not an observed association")
            for factor, tokens in values.items() for token in tokens)
        for token, label in pairs:
            _, system, rest = family(token)
            if system in {"icd10", "atc", "snomed", "sks_procedure", "npu"}:
                add(prototype, name, label, system, rest.split("//")[0])

    for n in range(95, 100):
        add("opera/interpretability/code_meaning.py", "MORPHOLOGY_ROLLUP",
            f"Morphology codes M{n}", "snomed", f"M{n}")

    # Literal code references in the outcome script. This is a code-identity
    # inventory, not validation of the full phenotype or its clinical naming.
    r_path = Path("opera/R/calculate_adverse_events_update.R")
    for number, line in enumerate((root / r_path).read_text().splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for code in re.findall(r'"\^?((?:NPU\d{5})|(?:[A-Z]\d{2})|(?:B[A-Z]{2}[A-Z0-9]*))"', line):
            system = "npu" if code.startswith("NPU") else (
                "sks_procedure" if len(code) > 3 or code[1:].isalpha() else "icd10")
            add(r_path, f"line {number}", "Endpoint code reference; phenotype requires separate review",
                system, code, "reference")
    for number in list(range(44)) + list(range(45, 80)) + list(range(81, 98)):
        add(r_path, "ICD10_CANCER_PREFIXES (generated)", "Any malignancy excluding C44 and C80",
            "icd10", f"C{number:02}")
    for code in [f"C{number}" for number in range(81, 97)] + ["D45", "D46", "D47"]:
        add(r_path, "ICD10_HEME_PREFIXES (generated)", "Haematological malignancy code set",
            "icd10", code)

    # Record the exclusions alongside positive molecular codes for an auditable
    # distinction between a result and the absence of any result.
    exclusions = ["FE16E0", "FE16E1", "FE16E2", "FE16E5", "FE16E9", "FE16EE",
                  "FK17D1", "FE13N0", "FE13N3", "FE13N7", "FE18U0", "FE18U3",
                  "FE18U5", "FE18U9", "FE18UB", "J08140", "J08141", "J3N8U0", "J3N8U2", "JFR8U0"]
    for code in exclusions:
        add("docs/DANISH_CODE_AUDIT_20261028.md", "Molecular exclusions",
            "Not included in positive-result concepts", "snomed", code, "exact")
    add("docs/DANISH_CODE_AUDIT_20261028.md", "Rejected former LDH mapping",
        "No assignment retained", "npu", "NPU19123", "exact")
    add("docs/DANISH_CODE_AUDIT_20261028.md", "Rejected former synthetic biopsy mapping",
        "No assignment retained", "sks_procedure", "KTSF")
    return pd.DataFrame(rows), config


def build_audit(assignments, sks, npu, pathology):
    sks_groups = {code: frame for code, frame in sks.groupby("code", sort=False)}
    npu_groups = {code: frame for code, frame in npu.groupby("npu_code", sort=False)}
    rows = []
    for item in assignments.to_dict("records"):
        system, code = item["system"], item["code"]
        out = {**item, "official_label": "", "unit": "", "valid_from": "", "valid_to": "",
               "public_identity": "NOT FOUND", "source": "", "notes": NOTES.get(code, ""),
               "study_vocabulary": "not checked (server vocabulary unavailable)"}
        if system == "snomed":
            hits = pathology.get(code, [])
            if hits:
                out.update(official_label=" | ".join(dict.fromkeys(label for label, _ in hits)),
                           source=" | ".join(dict.fromkeys(url for _, url in hits)),
                           public_identity="verified current public codebook; historical dates pending")
            elif item["match_mode"] == "prefix":
                descendants = [(c, label) for c, hits in pathology.items() if c.startswith(code)
                               for label, _ in hits]
                if descendants:
                    out.update(public_identity="mixed morphology prefix verified through descendants",
                               official_label=" | ".join(f"{c}: {label}" for c, label
                                                         in list(dict.fromkeys(descendants))[:4]),
                               source="https://www.patobank.dk/kodebog/?tag=E | "
                                      "https://www.patobank.dk/kodebog/?tag=D",
                               notes="Mixed code family; no single clinical diagnosis inferred.")
        elif system == "npu":
            frame = npu_groups.get(code)
            out["source"] = NPU_URL
            if frame is not None:
                row = frame.sort_values(["current_version", "change_date"]).iloc[-1]
                out.update(official_label=row["full_definition"], unit=row["unit"],
                           valid_from=row["effective_from"], valid_to=row["effective_to"],
                           public_identity="verified NPU definition" if str(row["active"]) == "1"
                           else "verified historical NPU definition")
        else:
            key = ("D" if system == "icd10" else "M" if system == "atc" else "") + code
            frame = sks_groups.get(key)
            out["source"] = SKS_URL
            if frame is not None:
                row = frame.sort_values(["valid_from", "change_date"]).iloc[-1]
                out.update(official_label=row["label"], valid_from=row["valid_from"],
                           valid_to=row["valid_to"], public_identity="verified SKS code"
                           if row["valid_to"] == "25000101" else "verified historical SKS code")
            elif item["match_mode"] != "exact":
                children = sks.loc[sks.code.str.startswith(key)]
                if len(children):
                    out.update(public_identity="prefix verified through descendants; no exact heading",
                               official_label=" | ".join(children.label.drop_duplicates().head(3)))
            if out["public_identity"] == "NOT FOUND" and system == "icd10" and (
                    "-" in item["assignment"] or "(generated)" in item["assignment"]):
                out.update(public_identity="unassigned category in numerical range",
                           notes="No official code at this range position. It matches no SKS records; "
                                 "this does not indicate missing study data.")
        if system == "npu" and code != "NPU19123":
            out["notes"] += (" Relative model bins are not clinical reference intervals. "
                             "Other NPU/local codes may measure the same analyte; coverage not established.")
        if code.startswith("BOQA0") and code != "BOQA0":
            out["notes"] = NOTES["BOQA0"]
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    sks = read_sks(args.sources / "SKScomplete.txt")
    npu = pd.read_csv(args.sources / "npu-codes-latest.csv", dtype=str, keep_default_na=False)
    pathology = {}
    for name, url in SOURCES.items():
        if name.endswith(".html"):
            table = PatobankTable()
            table.feed((args.sources / name).read_text())
            if not table.rows:
                raise ValueError(f"No Patobank rows parsed from {name}")
            for code, label in table.rows:
                pathology.setdefault(code, []).append((label, url))
    assignments, config = inventory(args.repo_root)
    audit = build_audit(assignments, sks, npu, pathology)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "DANISH_CODE_ASSIGNMENT_AUDIT_20261028.csv", index=False)
    unique = audit.drop_duplicates(["system", "code"])
    unique[["system", "code", "official_label", "unit", "valid_from", "valid_to",
            "public_identity", "source"]].to_csv(
                args.output_dir / "DANISH_VERIFIED_CODE_EXTRACT_20261028.csv", index=False)
    ledger = []
    for c in config["concepts"]:
        part = audit.loc[audit.assignment == f"concept:{c['name']}"]
        ledger.append({"concept": c["name"], "proposed_label": c["label"],
                       "token_prefixes": " | ".join(c["codes"]),
                       "match_mode": "exact" if c.get("exact_codes") else "prefix",
                       "code_identity_status": " | ".join(part.public_identity.unique()),
                       "source": " | ".join(part.source.unique()),
                       "study_vocabulary_status": "pending token coverage, dates and bin-boundary audit",
                       "notes": " | ".join(dict.fromkeys(part.notes[part.notes != ""]))})
    pd.DataFrame(ledger).to_csv(
        args.output_dir / "ATTRIBUTION_CONCEPT_MAPPING_REVIEW_20261028.csv", index=False)
    manifest = [{"file": name, "url": url, "bytes": (args.sources / name).stat().st_size,
                 "sha256": hashlib.sha256((args.sources / name).read_bytes()).hexdigest()}
                for name, url in SOURCES.items()]
    (args.output_dir / "DANISH_CODE_SOURCES_20261028.json").write_text(
        json.dumps({"retrieved_on": "2026-10-28", "sources": manifest,
                    "sks_records": len(sks), "npu_records": len(npu),
                    "pathology_codes": len(pathology)}, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(audit)} assignments; {len(unique)} distinct system/code pairs")
    print(unique.groupby("public_identity").size().to_string())
    missing = unique.loc[unique.public_identity == "NOT FOUND", ["system", "code", "assigned_label"]]
    if len(missing):
        print("Not found (requires interpretation, not automatic deletion):")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
