"""Synthetic-only input, manual execution and preservation checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_four_models as pipeline
import numpy as np
import pandas as pd


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = pipeline.paper_settings()
        x, y = pipeline.core.synthetic_fixture(settings, n=80, seed=20261109)
        scanner = np.empty(len(y), dtype=int)
        for outcome in (0, 1):
            selected = np.flatnonzero(y.to_numpy() == outcome)
            scanner[selected] = np.arange(len(selected)) % 2
        cls.payload = {"study_ids": list(x.index), "x": x.to_dict(orient="list"),
                       "y": y.astype(int).tolist(),
                       "columns": {key: list(value) for key, value in settings.model_columns().items()},
                       "plans": pipeline.core.make_split_plan(y, settings)}
        cls.metadata = pd.DataFrame({"study_id": list(x.index),
                                     "scanner": [pipeline.LEVELS[value] for value in scanner],
                                     "scanner_indicator": scanner})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "model_input.json"
        self.input.write_text(json.dumps(self.payload), encoding="utf-8")
        self.csv = self.root / "scanners.csv"
        self.metadata.to_csv(self.csv, index=False)

    def prepared(self):
        work = self.root / "private_work"
        pipeline.prepare(self.input, self.csv, work)
        return work

    def test_joint_inner_plan_and_original_outer_are_fixed(self):
        original = deepcopy(self.payload["plans"])
        ids, x, y, settings, columns, plans, cells = pipeline.assemble(self.payload, self.metadata)
        self.assertEqual(self.payload["plans"], original)
        self.assertEqual(len(plans), 100)
        for before, after in zip(original, plans):
            for field in ("repeat", "fold", "train", "test", "inner_seed"):
                self.assertEqual(before[field], after[field])
        self.assertEqual(tuple(columns), pipeline.MODELS)
        self.assertEqual(columns["CT"], settings.clinical + settings.t1)
        self.assertTrue(all(min(row["scanner_outcome_cells"][level].values()) > 0
                            for row in cells for level in pipeline.LEVELS))
        again = pipeline.assemble(self.payload, self.metadata)[5]
        self.assertEqual(plans, again)

    def test_metadata_join_uses_ids_not_row_order(self):
        forward = pipeline.assemble(self.payload, self.metadata)
        reverse = pipeline.assemble(self.payload, self.metadata.iloc[::-1])
        pd.testing.assert_frame_equal(forward[1], reverse[1])
        self.assertEqual(forward[5], reverse[5])

    def test_reject_missing_duplicate_unknown_or_inconsistent_scanner(self):
        variants = [self.metadata.iloc[:-1], pd.concat([self.metadata, self.metadata.iloc[:1]])]
        changed = self.metadata.copy()
        changed.loc[0, "scanner"] = "UNVERIFIED SCANNER"
        variants.append(changed)
        changed = self.metadata.copy()
        changed.loc[0, "scanner_indicator"] = 1 - changed.loc[0, "scanner_indicator"]
        variants.append(changed)
        for metadata in variants:
            with self.subTest(rows=len(metadata)):
                with self.assertRaises(ValueError):
                    pipeline.assemble(self.payload, metadata)

    def test_reject_scanner_outcome_separation(self):
        metadata = self.metadata.copy()
        metadata["scanner_indicator"] = self.payload["y"]
        metadata["scanner"] = [pipeline.LEVELS[value] for value in self.payload["y"]]
        with self.assertRaises(ValueError):
            pipeline.assemble(self.payload, metadata)

    def test_prepare_preserves_input_and_does_not_fit(self):
        original_hash = pipeline.digest(self.input)
        work = self.prepared()
        self.assertEqual(original_hash, pipeline.digest(self.input))
        self.assertFalse((work / "models").exists())
        self.assertEqual(pipeline.read(work / "preflight.json")["status"], "PASS")
        self.assertFalse(pipeline.read(work / "preflight.json")["model_fit_performed"])
        self.assertEqual(pipeline.verify_prepared(work)[1], self.payload["study_ids"])
        with self.assertRaises(FileExistsError):
            pipeline.prepare(self.input, self.csv, work)

    def test_changed_prepared_file_is_rejected(self):
        work = self.prepared()
        with (work / "protocol.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Frozen"):
            pipeline.verify_prepared(work)

    def test_changed_code_fingerprint_is_rejected(self):
        work = self.prepared()
        with patch.object(pipeline, "code_fingerprints", return_value=[]):
            with self.assertRaisesRegex(ValueError, "code changed"):
                pipeline.verify_prepared(work)

    def test_execution_is_opt_in_and_failure_is_preserved(self):
        work = self.prepared()
        plan = pipeline.fit_one(work, "C")
        self.assertEqual(plan["status"], "PLAN_ONLY_NOT_FITTED")
        self.assertFalse((work / "models").exists())
        with patch.object(pipeline, "fit_fold", side_effect=RuntimeError("synthetic intentional failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic intentional failure"):
                pipeline.fit_one(work, "C", execute=True)
        status = pipeline.read(work / "models/C/status.json")
        self.assertEqual(status["status"], "FAILED_PRESERVED")
        self.assertEqual(status["completed_outer_splits"], 0)
        with self.assertRaises(FileExistsError):
            pipeline.fit_one(work, "C", execute=True)
        self.assertFalse((work / "models/CT").exists())

    def test_output_boundaries_and_timeout(self):
        with self.assertRaises(ValueError):
            pipeline.new_private_output(pipeline.PACKAGE / "patient_results_not_allowed")
        with self.assertRaises(ValueError):
            pipeline.write(self.root / "outside.json", {}, boundary=self.root / "inside")
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pipeline.fit_one(self.root / "absent", "C", timeout=timeout)

    def test_optimized_python_cannot_disable_numerical_assertions(self):
        result = subprocess.run([sys.executable, "-O", "-B", str(Path(pipeline.__file__)), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Required numerical assertions are disabled", result.stderr)

    def test_penalty_mask_exactly_excludes_scanner(self):
        _, x, _, settings, columns, plans, _ = pipeline.assemble(self.payload, self.metadata)
        prep = pipeline.core.FoldPreprocessor(columns["CPT"], settings.categories).fit(x.iloc[plans[0]["train"]])
        kept = [field for field, keep in zip(prep.columns, prep.keep) if keep]
        mask = pipeline.mask_for(prep)
        self.assertEqual(mask[kept.index(pipeline.SCANNER)], 0)
        self.assertEqual(int((mask == 0).sum()), 1)

    def test_summary_rejects_incomplete_models_without_creating_output(self):
        work = self.prepared()
        summary = self.root / "summary"
        with self.assertRaises(FileNotFoundError):
            pipeline.summarize(work, summary)
        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
