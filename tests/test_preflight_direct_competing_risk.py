from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from opera.run.preflight_direct_competing_risk import _register_resolvers


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_resolves_inherited_version_interpolation(monkeypatch):
    fake_root = ROOT / "_preflight_paths_need_not_exist"
    monkeypatch.setenv("BONSAI_CONFIG_PATH", str(ROOT / "configs"))
    monkeypatch.setenv("BONSAI_MODELS", str(fake_root / "models"))
    monkeypatch.setenv("BONSAI_PROCESSED_DATA", str(fake_root / "processed"))
    monkeypatch.setenv("BONSAI_CHECKPOINT_ROOT", str(fake_root / "checkpoints"))
    monkeypatch.setenv("BONSAI_OUTCOMES_DIR", str(fake_root / "outcomes"))
    monkeypatch.setenv(
        "BONSAI_COHORT_MEMBERSHIP",
        str(fake_root / "processed" / "population_full.csv"),
    )
    OmegaConf.clear_resolver("version")
    _register_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT / "opera" / "configs"),
        version_base="1.2",
    ):
        cfg = compose(
            config_name="generated/direct_cr_architecture_shared_linear"
        )

    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["run_id"] == 0
    assert resolved["competing_risk"]["readout_mode"] == "shared"
