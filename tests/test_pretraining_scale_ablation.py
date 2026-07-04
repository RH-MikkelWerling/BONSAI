from types import SimpleNamespace

from opera.run.pretraining_scale_ablation import run_cell


def test_scale_ablation_threads_complete_outcome_contract(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "opera.run.pretraining_scale_ablation.subprocess.run",
        fake_run,
    )

    run_cell(
        checkpoint_name="small",
        checkpoint_path="/checkpoints/small.ckpt",
        encoder_source="pretrain",
        cohort="dlbcl",
        outcome="aki_30d",
        outcome_path="/data/aki.parquet",
        data_dir="/data",
        output_dir=tmp_path,
        base_config="opera/configs/finetune.yaml",
        n_hours_start_include=1,
        n_hours_end_include=720,
        registry_start_date="2010-01-01",
        eligibility_path="/data/aki_audit.parquet",
        competing_outcome_path="/data/mortality.parquet",
    )

    assert len(commands) == 2
    for command in commands:
        joined = " ".join(map(str, command))
        assert "labels.registry_start_date=2010-01-01" in joined
        assert "paths.eligibility=/data/aki_audit.parquet" in joined
        assert "paths.competing_outcome=/data/mortality.parquet" in joined
