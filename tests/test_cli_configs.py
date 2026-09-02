from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path(__file__).resolve().parents[1]

BONSAI_CONFIGS = [
    "daly_care_data",
    "pretrain",
    "pretrain_event_objective",
    "finetune",
]

OPERA_CONFIGS = [
    "contrastive",
    "contrastive_multicohort",
    "dapt",
    "evaluate",
    "finetune",
    "daly_care_pretrain",
    "hybrid",
    "joint_finetune",
    "mol",
    "survival_finetune",
    "daly_care_survival_finetune",
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
    monkeypatch.setenv("BONSAI_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv(
        "BONSAI_COHORT_MEMBERSHIP", str(tmp_path / "processed" / "population_full.csv")
    )
    monkeypatch.setenv("BONSAI_PREDICTIONS", str(tmp_path / "predictions"))


@pytest.mark.parametrize("config_name", BONSAI_CONFIGS)
def test_bonsai_hydra_configs_compose(config_environment, config_name):
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.2",
    ):
        cfg = compose(config_name=config_name)
    assert cfg is not None
    if config_name in {"pretrain", "pretrain_event_objective", "finetune"}:
        assert cfg.overwrite is False
    if config_name == "pretrain":
        assert cfg.paths.dataset_class.endswith("ARPretrainDataset")
        assert cfg.model.causal is True
        assert cfg.model.value_bin_vocab_size == 0
        assert cfg.model.value_embedding_mode == "legacy"
        assert cfg.training.value_regression_loss_weight == 0.0
    if config_name == "pretrain_event_objective":
        assert cfg.training.event_normalized_code_loss is True
        assert "RC_REASON//" in cfg.training.input_only_target_prefixes
        assert cfg.training.ignore_target_tokens == ["[SEP]"]
    if config_name == "daly_care_data":
        assert cfg.splits == ["train", "tuning"]
        assert cfg.numeric_value_mode == "continuous"
        assert dict(cfg.vocabulary_cutoff_date) == {
            "year": 2022,
            "month": 1,
            "day": 1,
        }


@pytest.mark.parametrize("config_name", OPERA_CONFIGS)
def test_opera_hydra_configs_compose(config_environment, config_name):
    with initialize_config_dir(
        config_dir=str(ROOT / "opera" / "configs"),
        version_base="1.2",
    ):
        cfg = compose(config_name=config_name)
    assert cfg is not None
    if config_name not in {"evaluate"}:
        assert cfg.overwrite is False
    if config_name == "survival_finetune":
        assert cfg.training.batch_sampling.type == "auto"
        assert cfg.training.eval_monitor_metric == "auto"
        assert cfg.seed == 42
    if config_name == "daly_care_survival_finetune":
        assert cfg.dataset == "daly_care"
        assert cfg.encoder_source == "pretrain"
        assert cfg.outcome == "overall_survival"
        assert cfg.training_mode == "cox_exact_cached"
        assert cfg.hardware.precision == "16-mixed"
        assert cfg.hardware.compile_mode is None
        assert cfg.cohort_fine_col is None
        assert cfg.cohort_fine_value is None
        assert cfg.paths.competing_outcome is None
        assert str(cfg.encoder_ckpt).endswith("daly_care_pretrain/best.ckpt")
    if config_name in {"daly_care_pretrain", "dapt"}:
        assert dict(cfg.training.cutoff_date) == {
            "year": 2022,
            "month": 1,
            "day": 1,
        }
    if config_name == "daly_care_pretrain":
        assert cfg.model.value_embedding_mode == "film"
        assert cfg.model.value_bin_vocab_size == 0
        assert cfg.model.abspos_encoding == "fourier"
        assert cfg.training.value_regression_loss_weight == 1.0
        assert cfg.model.hidden_size == 64
        assert cfg.model.num_layers == 4
        assert cfg.model.num_attention_heads == 4
        assert cfg.model.max_seqlen == 3372
        assert cfg.model.attn_type == "sdpa"
        assert cfg.hardware.precision == "16-mixed"
        assert cfg.hardware.compile_mode is None
        assert cfg.training.max_len == 3372
        assert cfg.training.batch_size == 128
        assert cfg.training.accumulate_grad_batches == 1
        assert cfg.training.epochs == 10
        assert cfg.training.learning_rate == pytest.approx(3e-4)
        assert cfg.training.scheduler_warmup_epochs == pytest.approx(0.1)
        assert cfg.training.truncation_strategy == "tail"
    if config_name == "generated/joint_opera_full_panel":
        assert cfg.dataset == "daly_care_joint_opera"
        assert "generated" not in cfg
    if config_name == "generated/multi_outcome_full_panel":
        assert cfg.dataset == "daly_care_multi_outcome"
        assert "generated" not in cfg


def test_survival_config_accepts_sweep_seed_override(config_environment):
    with initialize_config_dir(
        config_dir=str(ROOT / "opera" / "configs"),
        version_base="1.2",
    ):
        cfg = compose(config_name="survival_finetune", overrides=["seed=43"])

    assert cfg.seed == 43
