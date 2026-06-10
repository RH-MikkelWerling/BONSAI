"""Stable OPERA-facing imports for BONSAI internals.

OPERA intentionally builds on BONSAI, but collaborators may update BONSAI
module paths or constructor details independently.  Import shared BONSAI
objects through this module when touching OPERA code; if BONSAI moves a symbol,
the repair should usually happen here rather than across every experiment
script.
"""

from __future__ import annotations

from importlib import import_module


_SYMBOLS = {
    "BiGRU": ("bonsai.modules.networks.components.heads", "BiGRU"),
    "BonsaiEncoder": ("bonsai.modules.networks.bonsai_nets", "BonsaiEncoder"),
    "BonsaiFinetune": ("bonsai.modules.networks.bonsai_nets", "BonsaiFinetune"),
    "BonsaiPretrain": ("bonsai.modules.networks.bonsai_nets", "BonsaiPretrain"),
    "FinetuneDataset": ("bonsai.modules.datasets.FinetuneDataset", "FinetuneDataset"),
    "binarize_outcomes": ("bonsai.functional.outcomes", "binarize_outcomes"),
    "compute_abspos": ("bonsai.functional.features", "compute_abspos"),
    "dynamic_padding": ("bonsai.functional.collate", "dynamic_padding"),
    "filter_subject_data": ("bonsai.functional.subject_data", "filter_subject_data"),
    "split_and_binarize_outcomes": (
        "bonsai.functional.outcomes",
        "split_and_binarize_outcomes",
    ),
}


def __getattr__(name: str):
    try:
        module_name, symbol_name = _SYMBOLS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


__all__ = list(_SYMBOLS)
