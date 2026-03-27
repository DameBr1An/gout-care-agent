from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gout_agent.skills._runtime_loader import load_runtime_module


class LabReportSkillTests(unittest.TestCase):
    def test_parse_uploaded_lab_files_extracts_metrics_from_text_bytes(self) -> None:
        runtime = load_runtime_module("lab-report-skill")
        payload = runtime.parse_uploaded_lab_files(
            [
                {
                    "name": "lab-report.pdf",
                    "type": "application/pdf",
                    "bytes": "尿酸 430 umol/L 肌酐 85 umol/L eGFR 102".encode("utf-8"),
                }
            ]
        )
        self.assertIn("uric_acid", payload["metrics"])
        self.assertIn("creatinine", payload["metrics"])
        self.assertIn("egfr", payload["metrics"])
        self.assertEqual(payload["metrics"]["uric_acid"]["value"], 430.0)


if __name__ == "__main__":
    unittest.main()
