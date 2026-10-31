"""Run the router test files without pytest (the GPU server has none).

Supplies the small part of pytest these files use (parametrize, approx, raises,
fail, skip, module-level fixtures, and the tmp_path, monkeypatch and capsys
fixtures), runs every test and prints one line per failure and a summary. Uses
the real pytest instead when it is installed.

    .venv-v100/bin/python scripts/check_router_code.py
"""
from __future__ import annotations

import contextlib
import inspect
import importlib
import io
import itertools
import math
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = [
    "test_full_own_router", "test_stage8_integration", "test_router_smoke",
    "test_task_routed_router", "test_partial_pooling_router", "test_router_axis_separation",
    "test_paper_run_bundle", "test_paper_worker", "test_supervision_factorial", "test_paper_numbers",
]


class Skipped(Exception):
    pass


def _shim():
    module = types.ModuleType("pytest")

    class approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            return math.isclose(float(other), float(self.expected), rel_tol=self.rel, abs_tol=self.abs)

        def __repr__(self):
            return f"approx({self.expected})"

    @contextlib.contextmanager
    def raises(exception, match=None):
        try:
            yield
        except exception as error:
            if match is not None and not re.search(match, str(error)):
                raise AssertionError(f"{exception.__name__} raised, but {match!r} not in {error}") from error
            return
        raise AssertionError(f"DID NOT RAISE {exception.__name__}")

    def fail(message=""):
        raise AssertionError(message)

    def skip(message=""):
        raise Skipped(message)

    def fixture(function=None, **_):
        def mark(f):
            f._fixture = True
            return f
        return mark(function) if callable(function) else mark

    class _Mark:
        @staticmethod
        def parametrize(names, values, **_):
            def mark(f):
                f._parametrize = [*getattr(f, "_parametrize", []), (names, list(values))]
                return f
            return mark

        def __getattr__(self, _):
            return lambda *a, **k: (lambda f: f)

    module.approx, module.raises, module.fail, module.skip = approx, raises, fail, skip
    module.fixture, module.mark = fixture, _Mark()
    return module


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None, raising=True):
        if isinstance(target, str):
            module_name, attribute = target.rsplit(".", 1)
            target, value, name = importlib.import_module(module_name), name, attribute
        self._undo.append((setattr, target, name, getattr(target, name)))
        setattr(target, name, value)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, mapping.get(key, KeyError)))
        mapping[key] = value

    def setenv(self, name, value):
        import os
        self.setitem(os.environ, name, str(value))

    def delenv(self, name, raising=True):
        import os
        if name in os.environ:
            self._undo.append(("item", os.environ, name, os.environ[name]))
            del os.environ[name]

    def chdir(self, path):
        import os
        self._undo.append(("cwd", None, None, os.getcwd()))
        os.chdir(path)

    def undo(self):
        import os
        for kind, target, name, value in reversed(self._undo):
            if kind == "item":
                if value is KeyError:
                    target.pop(name, None)
                else:
                    target[name] = value
            elif kind == "cwd":
                os.chdir(value)
            else:
                setattr(target, name, value)


class CapSys:
    def readouterr(self):
        return types.SimpleNamespace(out="", err="")


def _cases(function):
    marks = getattr(function, "_parametrize", [])
    if not marks:
        return [({}, "")]
    axes = []
    for names, values in marks:
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        axes.append([dict(zip(names, v if len(names) > 1 else (v,))) for v in values])
    return [({k: v for part in combo for k, v in part.items()},
             "[" + "-".join(str(v) for part in combo for v in part.values()) + "]")
            for combo in itertools.product(*axes)]


def _call(function, module, params):
    stack, temporary = contextlib.ExitStack(), []

    def resolve(name):
        if name in params:
            return params[name]
        if name == "tmp_path":
            path = Path(tempfile.mkdtemp(prefix="router_check_"))
            temporary.append(path)
            return path
        if name == "monkeypatch":
            patch = MonkeyPatch()
            stack.callback(patch.undo)
            return patch
        if name == "capsys":
            return CapSys()
        fixture = getattr(module, name, None)
        if fixture is None or not getattr(fixture, "_fixture", False):
            raise LookupError(f"unknown fixture {name}")
        result = fixture(**{n: resolve(n) for n in inspect.signature(fixture).parameters})
        if inspect.isgenerator(result):
            value = next(result)
            stack.callback(lambda: next(result, None))
            return value
        return result

    try:
        with stack, contextlib.redirect_stdout(io.StringIO()):
            function(**{n: resolve(n) for n in inspect.signature(function).parameters})
    finally:
        for path in temporary:
            shutil.rmtree(path, ignore_errors=True)


def main():
    import os

    try:
        if os.environ.get("ROUTER_CHECK_WITHOUT_PYTEST"):
            raise ImportError
        import pytest  # noqa: F401
    except ImportError:
        sys.modules["pytest"] = _shim()
    else:
        sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "-W", "ignore",
                              *(str(REPO / "tests" / f"{name}.py") for name in FILES)]))
    sys.path[:0] = [str(REPO / "tests"), str(REPO)]
    import unittest

    passed, failed, skipped = 0, [], 0
    for name in FILES:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            failed.append((f"{name} (import)", traceback.format_exc(limit=3)))
            continue
        for attribute, function in list(vars(module).items()):
            if attribute.startswith("test_") and inspect.isfunction(function):
                for params, label in _cases(function):
                    try:
                        _call(function, module, params)
                        passed += 1
                    except Skipped:
                        skipped += 1
                    except Exception:  # noqa: BLE001
                        failed.append((f"{name}::{attribute}{label}", traceback.format_exc(limit=4)))
            elif inspect.isclass(function) and issubclass(function, unittest.TestCase) and function is not unittest.TestCase:
                result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(
                    unittest.defaultTestLoader.loadTestsFromTestCase(function))
                passed += result.testsRun - len(result.failures) - len(result.errors)
                failed += [(f"{name}::{case.id()}", text) for case, text in result.failures + result.errors]
    for label, text in failed:
        print(f"FAILED {label}\n{text}")
    print(f"{passed} passed, {len(failed)} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
