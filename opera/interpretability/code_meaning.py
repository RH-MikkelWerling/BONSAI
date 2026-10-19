"""From model tokens to clinical meaning.

Every token of the vocabulary is a register code with a source prefix
(``DIAGNOSIS//DC833``, ``MEDICATION//J01CA04``, ``LABTEST//NPU19673//BIN_3``,
``CONTACT_TYPE//ACUTE``). :class:`CodeDictionary` reads a token (or a rolled-up
concept, see :func:`opera.interpretability.attribution_summaries.rollup`) as

- its **source** (the register families of Figure S3, shared with the
  attribution summaries and the input restriction),
- its **code system** and normalised code (ICD-10 ``C83.3``, ATC ``J01CA04``,
  NPU ``NPU19673``, SNOMED ``M96803``, SKS procedure ``BOQA``),
- a **name**, from reviewed labels, then official code tables, then the
  built-in structure (ICD-10 chapters, ATC levels 1-2, ICD-O morphology
  ranges, clinical groups), then a readable form of the code,
- its **clinical group** (``Pneumonia``, ``Antibacterials``; the diagnosis
  groups use the codes of the outcome definitions) and **chapter** (ICD-10
  chapter, ATC anatomical group, SKS or SNOMED axis).

The dictionary also answers ``key in labels``, ``labels[key]`` and
``labels.get(key)``, so it can be passed wherever the attribution code takes
a mapping of labels. Laboratory value thirds (``LABTEST//<test>//low``) are
left to that code, which words them from the test's name.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

RESOURCES = Path(__file__).resolve().parents[1] / "resources" / "code_systems"
SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]")

# Token prefix -> (source, code system). The first matching prefix wins.
FAMILIES = (
    ("BG_", "Demographics", "demographics"),
    ("BACKGROUND//", "Demographics", "demographics"),
    ("SEX//", "Demographics", "demographics"),
    ("GENDER//", "Demographics", "demographics"),
    ("AGE//", "Demographics", "demographics"),
    ("DOB", "Demographics", "demographics"),  # ehr2meds birth event (bare code)
    ("DOD", "Demographics", "demographics"),  # ehr2meds death event; never before index
    ("DIAGNOSIS_TYPE//", "Diagnoses", "diagnosis_type"),
    ("DIAGNOSIS//", "Diagnoses", "icd10"),
    ("PROCEDURE_TYPE//", "Procedures", "procedure_type"),
    ("PROCEDURE//", "Procedures", "sks_procedure"),
    ("CONTACT_", "Hospital contacts", "contact"),
    ("PATIENT_TYPE", "Hospital contacts", "contact"),
    ("REGION//", "Hospital contacts", "contact"),
    # Raw contact namespaces of ehr2meds (configs/pretrain_event_objective.yaml).
    ("RC_", "Hospital contacts", "contact"),
    ("RD_", "Hospital contacts", "contact"),
    ("RMA_", "Hospital contacts", "contact"),
    ("RPO_", "Hospital contacts", "contact"),
    ("PATHOLOGY//", "Pathology", "snomed"),
    ("MEDICATION//", "Medication", "atc"),
    ("LABTEST//", "Laboratory", "npu"),
)
SOURCES = ("Demographics", "Diagnoses", "Procedures", "Hospital contacts", "Pathology",
           "Medication", "Laboratory")
OTHER_SOURCE = "Other"
DEMOGRAPHIC_PREFIXES = tuple(p for p, source, _ in FAMILIES if source == "Demographics")

ATC_PATTERN = re.compile(r"^[A-Z](\d{2}([A-Z]([A-Z](\d{2})?)?)?)?$")
NPU_DEFINITION = re.compile(r"^\s*(?P<system>[A-Za-z()]+)\s*[—–-]\s*(?P<component>[^;]+);")
NPU_SYSTEMS = {"U": "urine", "Csf": "CSF", "Fc": "faeces", "Pt": "patient",
               "Asc": "ascites", "Plr": "pleural fluid", "Sem": "semen"}
LAB_TEXT = {"NEG": "negative", "POS": "positive"}  # other text results: lower case
# Words for the attribute namespaces of contacts and demographics.
ATTRIBUTES = {"RC_ADM_TY": "Admission type", "RC_PRIO": "Priority",
              "RC_REASON": "Contact reason", "RC_REG": "Region", "RC_SPEC": "Specialty",
              "RC_TYPE": "Contact type", "RMA_PAT_TY": "Patient type",
              "CONTACT_TYPE": "Contact type", "CONTACT_SPECIALTY": "Specialty",
              "CONTACT_REASON": "Contact reason", "CONTACT_PRIORITY": "Priority",
              "DIAGNOSIS_TYPE": "Diagnosis type", "PROCEDURE_TYPE": "Procedure type",
              "PATIENT_TYPE": "Patient type", "REGION": "Region"}
SEX = {"F": "Female", "M": "Male", "K": "Female"}
# SKS main groups and SNOMED axes used in Danish pathology.
SKS_GROUPS = {"K": "Operations", "B": "Treatment and care", "U": "Examinations",
              "N": "Anaesthesia", "Z": "Miscellaneous procedures"}
SNOMED_AXES = {"M": "Morphology", "T": "Topography", "P": "Pathology procedure"}
# ICD-O-3 morphology (the SNOMED M codes of Danish pathology: M96803 = 9680/3).
MORPHOLOGY = ((9650, 9669, "Hodgkin lymphoma morphology"),
              (9590, 9729, "Lymphoma morphology"),
              (9731, 9734, "Plasma cell neoplasm morphology"),
              (9760, 9769, "Immunoproliferative disease morphology"),
              (9800, 9949, "Leukaemia morphology"),
              (9950, 9989, "Myeloid neoplasm morphology"))
MORPHOLOGY_ROLLUP = {"M95": "Lymphoma or other morphology", "M96": "Lymphoma morphology",
                     "M97": "Lymphoma or plasma cell morphology",
                     "M98": "Leukaemia morphology",
                     "M99": "Leukaemia or myeloid neoplasm morphology"}
NAMED = ("reviewed", "table", "built-in", "group")


def family(token: str) -> tuple[str, str, str]:
    """``(source, system, code after the prefix)`` of a token or concept."""
    token = str(token)
    if token in SPECIAL_TOKENS:
        return "Special", "special", token
    for prefix, source, system in FAMILIES:
        if token.startswith(prefix):
            rest = token[len(prefix):] if prefix.endswith("//") else token.partition("//")[2]
            return source, system, rest
    return OTHER_SOURCE, "other", token


def source_of(token: str) -> str:
    source = family(token)[0]
    return OTHER_SOURCE if source == "Special" else source


def normalise(system: str, code: str) -> str:
    """Code as looked up in tables: upper case, no dots, no SKS ``D`` on diagnoses."""
    code = str(code).strip().upper().replace(".", "").replace(" ", "")
    if system == "icd10" and len(code) >= 4 and code[0] == "D" and code[1].isalpha():
        code = code[1:]
    if system == "atc" and code.startswith("M") and ATC_PATTERN.match(code[1:] or "-"):
        code = code[1:]  # SKS writes medication codes as M + ATC
    return code


def display(system: str, code: str) -> str:
    """Code as written in the paper: ``C83.3``, ``J01CA04``, ``NPU19673``."""
    if system == "icd10":
        return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
    return code


def _ranges(entries):
    out = []
    for entry in entries:
        first, _, last = str(entry).partition("-")
        out.append((first.upper(), (last or first).upper()))
    return out


def _in_ranges(category, ranges):
    return any(first <= category[:len(first)] <= last for first, last in ranges)


def _npu_name(text):
    match = NPU_DEFINITION.match(str(text))
    if not match:
        return str(text).strip()
    component = match["component"].strip()
    system = NPU_SYSTEMS.get(match["system"])
    return f"{component} ({system})" if system else component


@lru_cache(maxsize=1)
def _builtin():
    chapters = pd.read_csv(RESOURCES / "icd10_chapters.csv", dtype=str)
    atc = pd.read_csv(RESOURCES / "atc_groups.csv", dtype=str)
    groups = yaml.safe_load((RESOURCES / "clinical_groups.yaml").read_text())
    return chapters, dict(zip(atc.code, atc.name)), groups


def read_table(spec: dict) -> dict[str, str]:
    """Normalised code -> name from one official table (csv, tsv, txt or Excel)."""
    path = Path(os.path.expandvars(str(spec["path"])))
    if not path.is_file():
        raise FileNotFoundError(f"Code table for {spec['system']} not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype=str, sheet_name=spec.get("sheet", 0))
    else:
        frame = pd.read_csv(path, dtype=str, sep=spec.get("sep"), engine="python",
                            encoding=spec.get("encoding", "utf-8"))
    code, label = spec.get("code", "code"), spec.get("label", "label")
    for column in (code, label):
        if column not in frame:
            raise KeyError(f"{path.name} has no column {column!r} (columns: {list(frame)})")
    frame = frame.dropna(subset=[code, label])
    names = frame[label].map(_npu_name) if spec["system"] == "npu" else frame[label].str.strip()
    return dict(zip(frame[code].map(lambda c: normalise(spec["system"], c)), names))


class CodeDictionary:
    """Names, groups and chapters for model tokens and rolled-up concepts."""

    def __init__(self, labels=None, tables=None, groups=None):
        chapters, self._atc, builtin_groups = _builtin()
        self._chapters = list(chapters.itertuples(index=False))
        self.labels = dict(labels or {})
        self._reviewed = {}
        for key, label in self.labels.items():
            _, system, rest = family(key)
            code, _, value = rest.partition("//")
            self._reviewed[(system, normalise(system, code), value.lower())] = label
        self.tables = {system: dict(names) for system, names in (tables or {}).items()}
        groups = groups if groups is not None else builtin_groups
        self._groups = {
            "icd10": [(name, _ranges(codes)) for name, codes in groups.get("diagnoses", {}).items()],
            "atc": [(name, codes) for name, codes in groups.get("medication", {}).items()],
            "sks_procedure": [(name, codes)
                              for name, codes in groups.get("procedures", {}).items()],
            "npu": [(name, codes) for name, codes in groups.get("laboratory", {}).items()],
        }

    @classmethod
    def from_config(cls, config: dict | None, *, reviewed: bool | None = None):
        """Reviewed labels (only when the mapping is reviewed) and official tables."""
        config = config or {}
        if reviewed is None:
            reviewed = bool(config.get("concept_mapping_reviewed"))
        settings = config.get("summary", {}) or {}
        labels = dict(settings.get("concept_labels") or {}) if reviewed else {}
        tables = {}
        for spec in config.get("code_tables") or []:
            tables.setdefault(spec["system"], {}).update(read_table(spec))
        groups = None
        if config.get("clinical_groups"):
            groups = yaml.safe_load(Path(os.path.expandvars(
                str(config["clinical_groups"]))).read_text())
        return cls(labels, tables, groups)

    # -- structure -------------------------------------------------------
    def _parts(self, token):
        source, system, rest = family(token)
        code, _, value = rest.partition("//")
        return source, system, normalise(system, code), value

    def group(self, token) -> str | None:
        """Clinical group of the code (the longest matching entry), if any."""
        _, system, code, _ = self._parts(token)
        best, length = None, 0
        for name, entries in self._groups.get(system, []):
            if system == "icd10":
                hit = len(code) >= 3 and _in_ranges(code[:3], entries)
                size = 3 if hit else 0
                for first, last in entries:  # an exact category beats a range
                    if hit and first == last and code.startswith(first):
                        size = len(first) + 1
            else:
                matches = [len(e) for e in map(str, entries) if code.startswith(e.upper())]
                hit, size = bool(matches), max(matches, default=0)
            if hit and size > length:
                best, length = name, size
        return best

    def chapter(self, token) -> str | None:
        """ICD-10 chapter, ATC anatomical group, SKS main group or SNOMED axis."""
        _, system, code, _ = self._parts(token)
        if system == "icd10" and len(code) >= 3:
            for row in self._chapters:
                if row.first <= code[:3] <= row.last:
                    return row.short
        if system == "atc" and not ATC_PATTERN.match(code):
            return SKS_GROUPS.get(code[:1])  # SKS code for hospital medication
        if system == "atc" and code[:1] in self._atc:
            return self._atc[code[:1]]
        if system == "sks_procedure":
            return SKS_GROUPS.get(code[:1])
        if system == "snomed":
            return SNOMED_AXES.get(code[:1])
        return None

    # -- names -----------------------------------------------------------
    def _builtin_name(self, system, code):
        if system == "atc":
            return self._atc.get(code)
        if system == "snomed" and code.startswith("M"):
            if len(code) <= 3:
                return MORPHOLOGY_ROLLUP.get(code)
            digits = code[1:5]
            if digits.isdigit():
                return next((name for first, last, name in MORPHOLOGY
                             if first <= int(digits) <= last), None)
        return None

    def _readable(self, source, system, code, value, token):
        if system == "icd10":
            return f"ICD-10 {display(system, code)}"
        if system == "atc":
            return f"ATC {code}" if ATC_PATTERN.match(code) else f"SKS {code}"
        if system == "sks_procedure":
            return f"SKS {code}"
        if system == "snomed":
            return f"SNOMED {code}"
        if system == "npu":
            return code
        if source == "Hospital contacts" or system in {"demographics", "diagnosis_type",
                                                       "procedure_type"}:
            token = str(token).removeprefix("BACKGROUND//")  # untimed codes of BONSAI
            if token in {"DOB", "DOD"}:
                return {"DOB": "Date of birth", "DOD": "Date of death"}[token]
            head, _, rest = token.partition("//")
            head = head.replace("BG_", "").replace("BACKGROUND", "").strip("_/")
            if head.upper() in {"SEX", "GENDER"} and rest.upper() in SEX:
                return SEX[rest.upper()]
            name = ATTRIBUTES.get(head, head.replace("_", " ").strip().capitalize())
            words = rest.replace("//", " ").replace("_", " ").lower()
            return f"{name} {words}".strip()
        return str(token)

    def _levels(self, system, code):
        """The code and its parents, finest first."""
        if system == "icd10":
            return [code[:k] for k in range(len(code), 2, -1)]
        if system == "atc":
            return [code[:k] for k in (7, 5, 4, 3, 1) if k <= len(code)] \
                if ATC_PATTERN.match(code) else [code[:k] for k in range(len(code), 1, -1)]
        if system == "sks_procedure":
            return [code[:k] for k in range(len(code), 1, -1)]
        if system == "snomed":
            return [code[:k] for k in range(len(code), 2, -1)]
        return [code]

    def resolve(self, token) -> tuple[str, str]:
        """``(name, where it came from)``: reviewed, table, built-in, group or code.

        The code is named at the finest level that has a name, from the code
        itself up to its chapter's subgroups (``J01CA04``, ``J01CA``, ``J01C``,
        ``J01``); at each level a reviewed label wins over a table, and a table
        over the built-in structure. Clinical groups come next, then the code.
        """
        token = str(token)
        if token in self.labels:
            return self.labels[token], "reviewed"
        source, system, code, value = self._parts(token)
        if system == "npu":
            name, origin = self._lab_name(code)
            text = (LAB_TEXT.get(value.upper(), value.lower().replace("_", " "))
                    if value else "")
            if text and not value.upper().startswith("BIN"):
                return f"{name} {text}", origin
            return name, origin
        table_system = "sks_procedure" if system == "atc" and not ATC_PATTERN.match(code) \
            else system  # SKS codes for hospital medication
        names = self.tables.get(table_system, {})
        for level in self._levels(system, code):
            if (system, level, value.lower()) in self._reviewed:
                return self._reviewed[(system, level, value.lower())], "reviewed"
            if level in names:
                return names[level], "table"
            builtin = self._builtin_name(system, level)
            if builtin:
                return builtin, "built-in"
        group = self.group(token)
        if group and system in {"icd10", "atc", "sks_procedure"}:
            return group, "group"
        return self._readable(source, system, code, value, token), "code"

    def _lab_name(self, test):
        if ("npu", test, "bin") in self._reviewed:
            return self._reviewed[("npu", test, "bin")], "reviewed"
        if test in self.tables.get("npu", {}):
            return self.tables["npu"][test], "table"
        group = self.group(f"LABTEST//{test}")
        return (group, "group") if group else (test, "code")

    def name(self, token) -> str:
        return self.resolve(token)[0]

    def describe(self, token) -> dict:
        """Everything the dictionary knows about one token."""
        source, system, code, value = self._parts(token)
        name, origin = self.resolve(token)
        return {"token": str(token), "source": source, "system": system,
                "code": display(system, code), "value": value or None, "name": name,
                "name_source": origin, "group": self.group(token),
                "chapter": self.chapter(token)}

    def table(self, tokens) -> pd.DataFrame:
        return pd.DataFrame([self.describe(t) for t in tokens])

    # -- the mapping interface used by the attribution code ----------------
    def _is_lab_third(self, key):
        source, system, _, value = self._parts(key)
        return system == "npu" and value.lower() in {"low", "mid", "high"} or (
            system == "npu" and value.upper().startswith("BIN_"))

    def __contains__(self, key):
        if key in self.labels:
            return True
        return not self._is_lab_third(key) and family(key)[0] not in {OTHER_SOURCE, "Special"}

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return self.name(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __bool__(self):
        return True

    def __len__(self):
        return len(self.labels)


__all__ = ["CodeDictionary", "DEMOGRAPHIC_PREFIXES", "FAMILIES", "OTHER_SOURCE", "SOURCES",
           "display", "family", "normalise", "read_table", "source_of"]
