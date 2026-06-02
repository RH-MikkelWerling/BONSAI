from pathlib import Path
from typing import Any, Dict, Tuple


def normalize_outcome_config(raw_outcomes: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize sweep outcome config.

    Supports the legacy list form:
      outcomes: [mortality_1y]

    and the preferred dict form:
      outcomes:
        mortality_1y:
          outcome_file: mortality.parquet
          n_hours_end_include: 8760
    """
    if isinstance(raw_outcomes, list):
        return {
            name: {
                "outcome_file": f"{name}.parquet",
                "n_hours_start_include": 1,
                "n_hours_end_include": None,
            }
            for name in raw_outcomes
        }
    return {
        name: {
            "outcome_file": cfg.get("outcome_file", f"{name}.parquet"),
            **cfg,
        }
        for name, cfg in raw_outcomes.items()
    }


def outcome_file_path(data_dir: str, outcome_name: str, outcome_cfg: Dict[str, Any]) -> str:
    """Resolve the source outcome parquet for a configured task."""
    raw = outcome_cfg.get("outcome_file", f"{outcome_name}.parquet")
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str(Path(data_dir) / "outcomes" / raw)


def parse_task_ref(task_ref: str) -> Tuple[str, str]:
    if ":" not in task_ref:
        raise ValueError(f"Task {task_ref!r} must be formatted as cohort:outcome")
    return tuple(task_ref.split(":", 1))
