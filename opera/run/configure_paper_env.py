"""Preview or atomically update paper path settings without exposing other env values."""

from __future__ import annotations

import argparse
import os
from io import StringIO
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream

CORE_KEYS = {
    "BONSAI_CONFIG_PATH",
    "BONSAI_MODELS",
    "BONSAI_PREDICTIONS",
    "BONSAI_CHECKPOINT_ROOT",
    "EHR2MEDS_OUTPUT",
    "BONSAI_PROCESSED_DATA",
    "BONSAI_RESULTS_ROOT",
    "BONSAI_COHORT_MEMBERSHIP",
    "BONSAI_OUTCOMES_DIR",
    "OPERA_RELEASE_DIR",
    "OPERA_SEQUENCE_DATA_DIR",
}


def update_env(path, settings, *, apply=False):
    path = Path(path).absolute()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Expected a regular, non-symlink env file")
    for key, value in settings.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not (
            key in CORE_KEYS or key.startswith("OPERA_PAPER_")
        ):
            raise ValueError(f"Unsupported paper environment key: {key}")
        if value is None or "\n" in value or "\r" in value or "\0" in value:
            raise ValueError(f"Expected a single-line value for {key}")
    raw = path.read_bytes() if path.exists() else b""
    original = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    seen = set()
    lines = []

    def assignment(key, value):
        value = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"{key}='{value}'{newline}"

    for binding in parse_stream(StringIO(original)):
        if binding.error:
            raise ValueError(
                "Existing env has invalid dotenv syntax; no settings were changed"
            )
        line, key = binding.original.string, binding.key
        if key not in settings:
            lines.append(line)
            continue
        if key in seen:
            raise ValueError(f"Duplicate {key}; resolve before updating")
        seen.add(key)
        lines.append(assignment(key, settings[key]))
    if lines and not lines[-1].endswith(("\r", "\n")):
        lines[-1] += newline
    for key, value in settings.items():
        if key not in seen:
            lines.append(assignment(key, value))
    replacement = "".join(lines).encode("utf-8")
    report = {
        "status": "unchanged" if raw == replacement else "preview",
        "env_file": str(path),
        "keys": sorted(settings),
        "values_redacted": True,
    }
    if not apply or raw == replacement:
        return report
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + f".backup.{stamp}")
        with backup.open("xb") as handle:
            handle.write(raw)
        shutil.copymode(path, backup)
    descriptor, temporary = tempfile.mkstemp(prefix=".paper-env-", dir=path.parent)
    try:
        os.fchmod(descriptor, path.stat().st_mode & 0o777 if path.exists() else 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        # Avoid overwriting edits made between preview construction and replacement.
        if (path.read_bytes() if path.exists() else b"") != raw:
            raise RuntimeError(
                "Env file changed during update; retry from current contents"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report.update(status="updated", backup=str(backup) if backup else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    settings = (
        dict(dotenv_values(args.settings_file, interpolate=False))
        if args.settings_file
        else {}
    )
    # Empty placeholders in an overlay must not erase configured server paths.
    skipped = []
    if args.env_file.exists():
        skipped = [key for key, value in settings.items() if value in (None, "")]
        settings = {key: value for key, value in settings.items() if key not in skipped}
    for entry in args.set:
        if "=" not in entry:
            raise ValueError("Use --set KEY=VALUE")
        key, value = entry.split("=", 1)
        settings[key] = value
    if not settings:
        raise ValueError("Supply --settings-file or --set")
    import json

    report = update_env(args.env_file, settings, apply=args.apply)
    report["skipped_empty_overlay_keys"] = skipped
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
