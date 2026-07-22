from pathlib import Path
from typing import Iterable


def resolve_meds_data_dir(path: Path, splits: Iterable[str]) -> Path:
    """Resolve current ehr2meds and legacy BONSAI event-shard layouts."""
    path = Path(path)
    split_names = tuple(splits)
    standard = path / "data"
    if standard.is_dir() and any((standard / split).is_dir() for split in split_names):
        return standard
    return path
