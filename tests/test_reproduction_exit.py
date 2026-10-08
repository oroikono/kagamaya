from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_recount", ROOT / "scripts/reproduce_findings.py")
RECOUNT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOUNT)


class ReproductionExitTests(unittest.TestCase):
    def passing_report(self):
        return {"status": "PASS", "checks": [{"passed": True}],
                "checks_total": 1, "checks_passed": 1, "checks_failed": 0}

    def test_pass_returns_zero(self):
        self.assertEqual(RECOUNT.report_exit_code(self.passing_report()), 0)

    def test_failed_check_returns_nonzero_even_with_pass_status(self):
        report = self.passing_report()
        report["checks"][0]["passed"] = False
        self.assertEqual(RECOUNT.report_exit_code(report), 1)

    def test_fail_status_returns_nonzero(self):
        report = self.passing_report(); report["status"] = "FAIL"
        self.assertEqual(RECOUNT.report_exit_code(report), 1)

    def test_missing_empty_and_inconsistent_reports_fail(self):
        for report in ({}, {"checks": []}, {**self.passing_report(), "checks_total": 2},
                       {**self.passing_report(), "checks_failed": False},
                       {**self.passing_report(), "checks": [{"passed": 1}]},
                       {**self.passing_report(), "checks": [None]}):
            with self.subTest(report=report):
                self.assertEqual(RECOUNT.report_exit_code(report), 1)

    def test_main_propagates_a_failed_registered_check(self):
        original_check = RECOUNT.check

        def fail_check(*args, **kwargs):
            row = original_check(*args, **kwargs)
            row["passed"] = False
            return row

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(RECOUNT, "OUT", Path(folder)), patch.object(RECOUNT, "check", fail_check), redirect_stdout(io.StringIO()):
                self.assertEqual(RECOUNT.main(), 1)


if __name__ == "__main__":
    unittest.main()
