"""Build, install, and validate the wheel outside the source checkout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="evidenceviewbench-wheel-") as raw:
        temp = Path(raw)
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(temp),
        )
        wheels = list(temp.glob("evidenceviewbench-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected exactly one wheel")
        run(sys.executable, "-m", "venv", str(temp / "venv"))
        python = temp / "venv/bin/python"
        pip = temp / "venv/bin/pip"
        run(str(pip), "install", "--no-deps", str(wheels[0]), cwd=temp)
        result = run(str(python), "-m", "evidenceviewbench.validation", cwd=temp)
        report = json.loads(result.stdout)
        if (
            report.get("valid") is not True
            or report.get("case_count") != 216
            or report.get("selection_observation_count") != 216
            or report.get("single_view_commit_checks") != 6912
        ):
            raise RuntimeError("installed-wheel validation did not pass")
        harness = run(
            str(python),
            "-c",
            (
                "from evidenceviewbench import load_two_d_public,load_two_d_owner,"
                "TwoDEvidenceEnv,TwoDAction,score_two_d_terminal;"
                "p=load_two_d_public();o=load_two_d_owner();r=p[0];"
                "e=TwoDEvidenceEnv(r,1000000);e.step(TwoDAction.acquire('V035'));"
                "t=e.step(TwoDAction.commit());"
                "label=next(x for x in o if x['origin_component']==r['origin_component'] "
                "and x['episode_id']==r['episode_id']);"
                "s=score_two_d_terminal(t,label);"
                "print(len(p),len(o),e.export_trace()['trace_sha256'],s.correct)"
            ),
            cwd=temp,
        )
        harness_fields = harness.stdout.strip().split()
        if len(harness_fields) != 4 or harness_fields[:2] != ["180", "180"]:
            raise RuntimeError("installed 2-D agent harness did not pass")
        print(
            json.dumps(
                {
                    "valid": True,
                    "wheel": wheels[0].name,
                    "wheel_bytes": wheels[0].stat().st_size,
                    "installed_case_count": report["case_count"],
                    "installed_observation_count": report["selection_observation_count"],
                    "installed_view_bank_count": report["view_bank_count"],
                    "installed_single_view_checks": report["single_view_commit_checks"],
                    "installed_two_d_public_rows": int(harness_fields[0]),
                    "installed_two_d_owner_rows": int(harness_fields[1]),
                    "installed_two_d_trace_sha256": harness_fields[2],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
