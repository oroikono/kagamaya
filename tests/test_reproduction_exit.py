from __future__ import annotations

import importlib.util
import io
import json
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

    def test_runtime_error_is_not_a_pass_even_with_successful_counters(self):
        report = self.passing_report()
        report["execution_status"] = "ERROR"
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
            report = json.loads((Path(folder) / "code_crosscheck.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["execution_status"], "COMPLETED")
            self.assertEqual(report["checks_total"], 33)
            self.assertEqual(report["checks_failed"], 33)

    def test_current_pass_is_invalidated_before_input_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            report_path = out / "code_crosscheck.json"
            report_path.write_text(json.dumps(self.passing_report()))

            def interrupted_recount():
                current = json.loads(report_path.read_text())
                self.assertEqual(current["status"], "RUNNING")
                self.assertEqual(current["execution_status"], "RUNNING")
                self.assertEqual(RECOUNT.report_exit_code(current), 1)
                raise KeyboardInterrupt("synthetic interruption")

            with patch.object(RECOUNT, "OUT", out), patch.object(RECOUNT, "recompute_report", interrupted_recount):
                with self.assertRaises(KeyboardInterrupt):
                    RECOUNT.main()
            current = json.loads(report_path.read_text())
            self.assertEqual(current["status"], "RUNNING")
            self.assertIsNone(current["finished_at"])

    def test_atomic_receipt_failure_does_not_publish_partial_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "code_crosscheck.json"
            path.write_text('{"status":"RUNNING"}\n')
            with patch.object(RECOUNT.os, "replace", side_effect=OSError("synthetic write failure")):
                with self.assertRaises(OSError):
                    RECOUNT.atomic_report(path, {"status": "FAIL"})
            self.assertEqual(json.loads(path.read_text()), {"status": "RUNNING"})
            self.assertEqual(list(Path(folder).glob(".code_crosscheck-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
