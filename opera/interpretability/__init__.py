"""Interpretability utilities for OPERA."""

from opera.interpretability.discordance import (
    characterize_discordance,
    find_discordant_patients,
    validate_discordant_features,
)
from opera.interpretability.integrated_gradients import (
    IntegratedGradientsExplainer,
    aggregate_ig_population,
    compute_ig,
)

__all__ = [
    "IntegratedGradientsExplainer",
    "aggregate_ig_population",
    "characterize_discordance",
    "compute_ig",
    "find_discordant_patients",
    "validate_discordant_features",
]
