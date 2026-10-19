import json
import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_pretrain_hidden256.sh"


def test_resume_reuses_saved_run_without_architecture_environment(tmp_path):
    run = tmp_path / "existing run"
    (run / "version_0").mkdir(parents=True)
    (run / "hydra").mkdir()
    (run / "version_0/last.ckpt").touch()
    (run / "hydra/config.yaml").write_text("model: {hidden_size: 256}\n")
    fake = tmp_path / "python-stub"
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = os.environ.copy()
    for key in ("BONSAI_256_HEADS", "BONSAI_256_LAYERS", "BONSAI_256_ARCHITECTURE_CONFIRMED"):
        env.pop(key, None)
    env.update(BONSAI_256_RUN_DIR=str(run), OPERA_PYTHON=str(fake))
    result = subprocess.run(["bash", str(SCRIPT), "resume"], env=env, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout.splitlines()[-1])
    assert args[:2] == ["-m", "bonsai.run.pretrain"]
    assert str(run / "hydra") in args
    assert f"paths.ckpt_path={run}/version_0/last.ckpt" in args
    assert "training.epochs=10" in args  # every encoder: 10 epochs
    assert not any("model.hidden_size" in arg for arg in args)


def test_resume_missing_checkpoint_fails_instead_of_starting_new_run(tmp_path):
    env = os.environ | {"BONSAI_256_RUN_DIR": str(tmp_path / "absent")}
    result = subprocess.run(["bash", str(SCRIPT), "resume"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Cannot resume: missing" in result.stderr
