from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path(__file__).resolve().parents[1]

BONSAI_CONFIGS = [
    "pretrain",
    "finetune",
]

OPERA_CONFIGS = [
    "contrastive",
    "contrastive_multicohort",
    "dapt",
    "evaluate",
    "finetune",
    "hematology_pretrain",
    "hybrid",
    "joint_finetune",
    "mol",
    "survival_finetune",
    "generated/joint_opera_full_panel",
    "generated/multi_outcome_full_panel",
]


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def config_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("BONSAI_CONFIG_PATH", str(ROOT / "configs"))
    monkeypatch.setenv("BONSAI_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("BONSAI_PROCESSED_DATA", str(tmp_path / "processed"))
    monkeypatch.setenv("BONSAI_PREDICTIONS", str(tmp_path / "predictions"))


@pytest.mark.parametrize("config_name", BONSAI_CONFIGS)
def test_bonsai_hydra_configs_compose(config_environment, config_name):
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.2",
    ):
        cfg = compose(config_name=config_name)
    assert cfg is not None
    if config_name == "pretrain":
        assert cfg.paths.dataset_class.endswith("ARPretrainDataset")
        assert cfg.model.causal is True


@pytest.mark.parametrize("config_name", OPERA_CONFIGS)
def test_opera_hydra_configs_compose(config_environment, config_name):
    with initialize_config_dir(
        config_dir=str(ROOT / "opera" / "configs"),
        version_base="1.2",
    ):
        cfg = compose(config_name=config_name)
    assert cfg is not None
    if config_name == "survival_finetune":
        assert cfg.training.batch_sampling.type == "auto"
