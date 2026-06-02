import importlib.util
import tomllib
from pathlib import Path


def test_bonsai_and_opera_packages_are_discoverable():
    assert importlib.util.find_spec("bonsai") is not None
    assert importlib.util.find_spec("opera") is not None


def test_pyproject_packages_bonsai_and_opera():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "bonsai.*" in include
    assert "opera.*" in include
