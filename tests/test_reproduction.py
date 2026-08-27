from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class ReleaseReproductionTest(unittest.TestCase):
    def test_all_claim_checks_pass(self) -> None:
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "scripts/reproduce_findings.py"], cwd=root, check=True)
        report = json.loads((root / "output/reproduction/code_crosscheck.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks_total"], 33)
        self.assertEqual(report["checks_failed"], 0)


if __name__ == "__main__":
    unittest.main()

