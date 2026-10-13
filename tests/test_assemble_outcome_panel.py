"""Self-contained tests; runnable without the full model test dependencies."""

import tempfile
import unittest
from pathlib import Path

import yaml

from opera.run.assemble_outcome_panel import assemble


class AssembleOutcomePanelTest(unittest.TestCase):
    def test_copies_exact_inventory_and_preserves_sources(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            base, second, composites = (root / name for name in ("base", "second", "composites"))
            for folder in (base, second, composites):
                folder.mkdir()
            (base / "one.parquet").write_bytes(b"base")
            (base / "one__audit.parquet").write_bytes(b"audit")
            (second / "two.parquet").write_bytes(b"second")
            (composites / "three.parquet").write_bytes(b"composite")
            (base / "coverage_gated_outcomes.csv").write_text("outcome\none\n")
            (base / "coverage_gate_provenance.txt").write_text("smr=2018-05-15\n")
            registry = root / "registry.yaml"
            registry.write_text(yaml.safe_dump({"outcomes": ["one", "two", "three"]}))
            target = root / "panel"

            result = assemble(base, second, composites, target, registry)

            self.assertEqual(result["outcomes"], 3)
            self.assertEqual((target / "two.parquet").read_bytes(), b"second")
            self.assertFalse((target / "one__audit.parquet").exists())
            self.assertTrue((base / "one.parquet").exists())
            self.assertTrue((target / "assembly_manifest_INTERNAL.json").exists())

    def test_rejects_duplicate_without_creating_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            base, second, composites = (root / name for name in ("base", "second", "composites"))
            for folder in (base, second, composites):
                folder.mkdir()
            (base / "one.parquet").write_bytes(b"base")
            (second / "one.parquet").write_bytes(b"second")
            (base / "coverage_gated_outcomes.csv").write_text("outcome\none\n")
            registry = root / "registry.yaml"
            registry.write_text(yaml.safe_dump({"outcomes": ["one"]}))
            target = root / "panel"

            with self.assertRaisesRegex(ValueError, "Duplicate outcome"):
                assemble(base, second, composites, target, registry)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
