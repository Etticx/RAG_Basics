"""Offline checks for evidence capture, metric routing and report integrity."""

import asyncio
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate_rag import load_cases, rescore_saved, run_evaluation, safe_error, save_results, score_answer
from evaluation_report import METRICS, build_graphs, write_report
from main import answer_question


class FakeMetric:
    def __init__(self, value=.8, failure=False):
        self.value, self.failure, self.calls = value, failure, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise RuntimeError("Simulated API failure")
        return SimpleNamespace(value=self.value)


class EvaluationTests(unittest.TestCase):
    def row(self, reference=None):
        return dict(question="What criteria?", answer="Criterion A.",
                    contexts=["Criterion A."], sources=[], reference=reference, notes={})

    def test_evidence_from_same_chain_invocation(self):
        calls = []
        class Chain:
            def invoke(self, question):
                calls.append(question)
                return {"answer": "A", "context": [
                    SimpleNamespace(page_content="Evidence", metadata={"page": 4})]}
        row = answer_question(Chain(), "Question")
        self.assertEqual(calls, ["Question"])
        self.assertEqual(row["contexts"], ["Evidence"])
        self.assertEqual(row["sources"], [{"page": 4}])

    def test_without_reference_only_reference_free_metrics_run(self):
        metrics = {name: FakeMetric() for name in METRICS}
        row = asyncio.run(score_answer(self.row(), metrics))
        self.assertEqual(row["faithfulness"], .8)
        self.assertIsNone(row["context_precision"])
        self.assertIsNone(row["context_recall"])
        self.assertEqual(metrics["context_recall"].calls, [])

    def test_reference_and_actual_context_are_routed_to_retrieval_metrics(self):
        metrics = {name: FakeMetric() for name in METRICS}
        row = asyncio.run(score_answer(self.row("Criterion A."), metrics))
        for name in ("context_precision", "context_recall"):
            self.assertEqual(metrics[name].calls, [{
                "user_input": row["question"], "reference": "Criterion A.",
                "retrieved_contexts": row["contexts"],
            }])

    def test_failed_invalid_and_negative_scores(self):
        metrics = {name: FakeMetric() for name in METRICS}
        metrics["faithfulness"] = FakeMetric(failure=True)
        metrics["answer_relevancy"] = FakeMetric(-.2)
        metrics["context_precision"] = FakeMetric(float("nan"))
        row = asyncio.run(score_answer(self.row("A"), metrics))
        self.assertIsNone(row["faithfulness"])
        self.assertIsNone(row["context_precision"])
        self.assertEqual(row["answer_relevancy"], -.2)
        self.assertEqual(row["context_recall"], .8)
        for svg in build_graphs({"rows": [row]}).values():
            ET.fromstring(svg)

    def test_graphs_exclude_missing_scores_and_report_escapes_evidence(self):
        first = self.row()
        first.update(faithfulness=.8, answer="<script>alert(1)</script>")
        second = self.row()
        second.update(faithfulness=None)
        payload = {"rows": [first, second], "created_at": "offline-test"}
        graphs = build_graphs(payload)
        self.assertIn("0.800 (n=1/2)", graphs["metric_averages.svg"])
        self.assertNotIn("Context recall", graphs["metric_averages.svg"])
        for svg in graphs.values():
            ET.fromstring(svg)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save_results(payload, output)
            write_report(payload, output)
            report = (output / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertEqual(json.loads((output / "results.json").read_text())["rows"], payload["rows"])

    def test_no_successful_scores_produces_no_fake_graph(self):
        self.assertEqual(build_graphs({"rows": [self.row()]}), {})

    def test_case_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text('[{"question": "", "reference": null}]')
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_scoring_uses_answers_captured_in_original_environment(self):
        metrics = {name: FakeMetric() for name in METRICS}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(model="baseline-model", judge_model="judge-model",
                                   rag_python=Path("original-venv/python.exe"),
                                   cases=output / "cases.json", limit=1)
            cases = [dict(question="What criteria?", reference="Criterion A.")]
            args.cases.write_text(json.dumps(cases))
            def capture(command, check):
                captured = self.row("Criterion A.")
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps([captured]))
            with patch("evaluate_rag.create_metrics", return_value=metrics), \
                    patch("evaluate_rag.subprocess.run", side_effect=capture) as dispatch:
                asyncio.run(run_evaluation(args, cases, output))
            command = dispatch.call_args.args[0]
            self.assertEqual(command[0], str(args.rag_python))
            self.assertEqual(command[command.index("--model") + 1], "baseline-model")
            payload = json.loads((output / "results.json").read_text())
            self.assertEqual(payload["rows"][0]["contexts"], ["Criterion A."])
            self.assertEqual(payload["rows"][0]["faithfulness"], .8)
            self.assertTrue((output / "metric_averages.svg").exists())

    def test_rescore_reuses_saved_evidence_without_running_generator(self):
        metrics = {name: FakeMetric() for name in METRICS}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "original.json"
            row = self.row()
            row["notes"]["faithfulness"] = "Failed: old synchronous-client error."
            payload = dict(rows=[row], model="unchanged-baseline", judge_model="judge",
                           created_at="original-time")
            source.write_text(json.dumps(payload))
            before = source.read_bytes()
            args = SimpleNamespace(rescore=source, limit=1, judge_model=None)
            output = Path(folder) / "rescored"
            with patch("evaluate_rag.create_metrics", return_value=metrics), \
                    patch("evaluate_rag.subprocess.run") as generator:
                asyncio.run(rescore_saved(args, output))
            generator.assert_not_called()
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["model"], "unchanged-baseline")
            self.assertEqual(result["rows"][0]["contexts"], row["contexts"])
            self.assertNotIn("faithfulness", result["rows"][0]["notes"])
            self.assertEqual(source.read_bytes(), before)

    def test_diagnostic_message_redacts_api_keys(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "private-test-key"}):
            detail = safe_error(TypeError("Async client required; key=private-test-key"))
        self.assertIn("Async client required", detail)
        self.assertNotIn("private-test-key", detail)
        self.assertIn("[REDACTED]", detail)


if __name__ == "__main__":
    unittest.main()
