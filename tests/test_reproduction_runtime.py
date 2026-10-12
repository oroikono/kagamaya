from __future__ import annotations

from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from scripts import reproduce_findings as RECOUNT


ROOT = Path(__file__).resolve().parents[1]
ARCHIVAL = ROOT / "output/audits/doomsday_v19_code_crosscheck_20260827/code_crosscheck.json"


class ReproductionRuntimeTests(unittest.TestCase):
    def run_recount(self, out, paths=None):
        with patch.object(RECOUNT, "OUT", out), patch.object(RECOUNT, "PATHS", paths or RECOUNT.PATHS), redirect_stdout(io.StringIO()):
            code = RECOUNT.main()
        return code, json.loads((out / "code_crosscheck.json").read_text())

    def assert_finished_attempt(self, report):
        self.assertEqual(report["receipt_schema_version"], "kagamaya.recount_attempt.v1")
        self.assertEqual(str(uuid.UUID(report["attempt_id"])), report["attempt_id"])
        start = datetime.datetime.fromisoformat(report["started_at"])
        finish = datetime.datetime.fromisoformat(report["finished_at"])
        self.assertIsNotNone(start.tzinfo)
        self.assertIsNotNone(finish.tzinfo)
        self.assertGreaterEqual(finish, start)
        self.assertEqual(report["environment"]["python_version"], RECOUNT.platform.python_version())
        source = report["source_identity"]
        self.assertEqual(source["recount_sha256"], RECOUNT.sha(Path(RECOUNT.__file__)))
        self.assertEqual(source["manifest_sha256"], RECOUNT.sha(ROOT / "MANIFEST.sha256"))
        self.assertIs(source["manifest_verified_by_recount"], False)

    def test_success_keeps_archived_scientific_checks_and_derived_values(self):
        frozen_bytes = ARCHIVAL.read_bytes()
        historical = json.loads(frozen_bytes)
        with tempfile.TemporaryDirectory() as folder:
            code, report = self.run_recount(Path(folder))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["execution_status"], "COMPLETED")
        self.assertEqual(report["checks_total"], 33)
        self.assertEqual(report["checks"], historical["checks"])
        self.assertEqual(report["derived"], historical["derived"])
        self.assertEqual(ARCHIVAL.read_bytes(), frozen_bytes)
        self.assert_finished_attempt(report)

    def test_missing_and_malformed_inputs_replace_an_earlier_real_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "reproduction"
            code, successful = self.run_recount(out)
            self.assertEqual(code, 0)
            previous_ids = {successful["attempt_id"]}
            fixtures = (
                ("missing_manuscript", "manuscript", None, "FileNotFoundError"),
                ("missing_jsonl", "worked_cases", None, "FileNotFoundError"),
                ("malformed_jsonl", "worked_cases", "{invalid json\n", "JSONDecodeError"),
                ("malformed_json", "quadrature", "{invalid json\n", "JSONDecodeError"),
            )
            for name, label, content, error_type in fixtures:
                with self.subTest(name=name):
                    # Seed the actual earlier success to reproduce independent
                    # readers seeing an old PASS before a failed new invocation.
                    (out / "code_crosscheck.json").write_text(json.dumps(successful))
                    broken = root / name
                    if content is not None:
                        broken.write_text(content)
                    paths = {**RECOUNT.PATHS, label: broken}
                    code, failed = self.run_recount(out, paths)
                    self.assertEqual(code, 1)
                    self.assertEqual(RECOUNT.report_exit_code(failed), 1)
                    self.assertEqual(failed["status"], "FAIL")
                    self.assertEqual(failed["execution_status"], "ERROR")
                    self.assertEqual(failed["execution_error"]["type"], error_type)
                    self.assertEqual(failed["checks"], [])
                    self.assertEqual((failed["checks_total"], failed["checks_passed"],
                                      failed["checks_failed"]), (0, 0, 0))
                    self.assertNotIn(failed["attempt_id"], previous_ids)
                    previous_ids.add(failed["attempt_id"])
                    self.assert_finished_attempt(failed)

    def test_input_invariants_do_not_depend_on_assertions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = root / "quadrature.json"
            bad.write_text('{"rows": []}\n')
            code, failed = self.run_recount(root / "reproduction", {**RECOUNT.PATHS, "quadrature": bad})
            self.assertEqual(code, 1)
            self.assertEqual(failed["execution_status"], "ERROR")
            self.assertEqual(failed["execution_error"]["type"], "ValueError")
            self.assertIn("5760", failed["execution_error"]["message"])
            markdown = root / "input.md"
            markdown.write_text("{}\n")
            for loader in (RECOUNT.load_json, RECOUNT.load_jsonl):
                with self.assertRaisesRegex(ValueError, "Markdown forbidden"):
                    loader(markdown)


if __name__ == "__main__":
    unittest.main()
