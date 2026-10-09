"""Run thesis questions through RAG, score the answers, then plot real results.

Usage: python evaluate_rag.py --limit 1
       python evaluate_rag.py --plot-only eval_outputs/RUN/results.json
"""

import argparse
import asyncio
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evaluation_report import METRICS, write_report
from main import BASE_DIR, DEFAULT_MODEL

EVAL_PYTHON = BASE_DIR / ".eval-venv" / "Scripts" / "python.exe"
RAG_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a non-empty JSON list.")
    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {i} must be an object.")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {i} needs a non-empty question.")
        reference = case.get("reference")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(f"Case {i}: reference must be text or null.")
    return cases


def create_metrics(judge_model, embedding_model):
    import httpx
    from google import genai
    from openai import AsyncOpenAI
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
    )
    from evaluation_transport import PacedAsyncTransport

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env.")
    # This is Google's Gemini endpoint; no OpenAI API key is used.
    # Ragas documents this route as a workaround for Gemini instructor issues.
    judge = llm_factory(
        judge_model,
        provider="openai",
        client=AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=90,
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=PacedAsyncTransport(
                    interval=float(os.getenv("RAG_EVAL_REQUEST_INTERVAL", "13"))
                ), timeout=90,
            ),
        ),
        temperature=0,
    )
    embeddings = GoogleEmbeddings(
        client=genai.Client(api_key=api_key), model=embedding_model
    )
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


def safe_error(error):
    message = str(error)
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"AIza[\w-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)(api_key|key)=([^\s&]+)", r"\1=[REDACTED]", message)
    return f"{type(error).__name__}: {message[:500]}"


async def score_answer(row, metrics):
    # Step 8: metrics plug in HERE, after generation, using its saved evidence.
    question, answer, contexts = row["question"], row["answer"], row["contexts"]
    reference = row.get("reference")
    inputs = {
        "faithfulness": dict(user_input=question, response=answer,
                             retrieved_contexts=contexts),
        "answer_relevancy": dict(user_input=question, response=answer),
    }
    if reference and reference.strip():
        inputs.update({
            "context_precision": dict(user_input=question, reference=reference,
                                      retrieved_contexts=contexts),
            "context_recall": dict(user_input=question, reference=reference,
                                   retrieved_contexts=contexts),
        })
    # A skipped or failed metric stays null; it is never silently scored as zero.
    for name in METRICS:
        row["notes"].pop(name, None)
        row[name] = None
        if name not in inputs:
            row["notes"][name] = "Skipped: no verified reference answer."
            continue
        try:
            result = await metrics[name].ascore(**inputs[name])
            value = float(result.value)
            lower_bound = -1 if name == "answer_relevancy" else 0
            if not math.isfinite(value) or not lower_bound <= value <= 1:
                raise ValueError("Metric returned an invalid score.")
            row[name] = value
        except Exception as error:
            detail = safe_error(error)
            row["notes"][name] = f"Failed: {detail}"
            print(f"  {name} failed ({detail}); continuing.")
    return row


def save_results(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["question", "reference", *METRICS]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["rows"])


async def run_evaluation(args, cases, output):
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    model = args.model or os.getenv("RAG_MODEL", DEFAULT_MODEL)
    judge_model = args.judge_model or os.getenv("RAG_EVAL_MODEL", model)
    embedding_model = os.getenv("RAG_EVAL_EMBEDDING_MODEL", "gemini-embedding-001")
    metrics = create_metrics(judge_model, embedding_model)
    # Capture answers in the ORIGINAL RAG environment. The evaluator's compatible
    # LangChain version must not alter the baseline's loading/retrieval/generation.
    output.mkdir(parents=True, exist_ok=True)
    captured_path = output / "captured_answers.json"
    command = [str(args.rag_python), str(BASE_DIR / "capture_rag.py"),
               "--cases", str(args.cases.resolve()), "--output", str(captured_path.resolve()),
               "--model", model]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True)
    captured = json.loads(captured_path.read_text(encoding="utf-8"))
    payload = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "model": model,
        "judge_model": judge_model,
        "evaluation_embedding_model": embedding_model,
        "cases_file": str(args.cases.resolve()),
        "rows": [],
    }
    await score_rows(payload, captured, metrics, output)


async def score_rows(payload, rows, metrics, output):
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] Scoring: {row['question']}")
        if row.get("answer") is not None:
            await score_answer(row, metrics)
        else:
            row.update(dict.fromkeys(METRICS))
        payload["rows"].append(row)
        # Checkpoint answers and scores so an interrupted run keeps completed work.
        save_results(payload, output)
    write_report(payload, output)
    if not any(row.get(name) is not None for row in payload["rows"] for name in METRICS):
        raise RuntimeError(f"No metric succeeded. Inspect {output / 'results.json'}")


async def rescore_saved(args, output):
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    source = json.loads(args.rescore.read_text(encoding="utf-8-sig"))
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("--rescore requires a results.json containing saved rows.")
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        if not isinstance(row.get("question"), str) or not isinstance(row.get("contexts"), list):
            raise ValueError("Each saved row needs question text and its original contexts.")
        row.setdefault("notes", {})
    judge_model = (args.judge_model or os.getenv("RAG_EVAL_MODEL")
                   or source.get("judge_model") or source.get("model", DEFAULT_MODEL))
    embedding_model = (os.getenv("RAG_EVAL_EMBEDDING_MODEL")
                       or source.get("evaluation_embedding_model", "gemini-embedding-001"))
    metrics = create_metrics(judge_model, embedding_model)
    payload = {**source, "rows": [],
               "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
               "judge_model": judge_model, "evaluation_embedding_model": embedding_model,
               "rescored_from": str(args.rescore.resolve()),
               "original_created_at": source.get("created_at")}
    await score_rows(payload, rows, metrics, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=BASE_DIR / "eval_cases.json")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N cases.")
    parser.add_argument("--model", help="Override the answer-generation Gemini model.")
    parser.add_argument("--judge-model", help="Override the evaluation Gemini model.")
    parser.add_argument("--rag-python", type=Path, default=RAG_PYTHON,
                        help="Python in the original RAG environment (default: venv).")
    parser.add_argument("--output", type=Path, help="Directory to save this run.")
    saved = parser.add_mutually_exclusive_group()
    saved.add_argument("--plot-only", type=Path,
                       help="Replot saved results without making API calls.")
    saved.add_argument("--rescore", type=Path,
                       help="Score saved results.json again without rerunning the baseline RAG.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.plot_only:
        payload = json.loads(args.plot_only.read_text(encoding="utf-8-sig"))
        output = args.output or args.plot_only.parent
        write_report(payload, output)
    elif args.rescore:
        if args.model:
            parser.error("--model cannot change a saved answer; use --judge-model for rescoring.")
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        output = args.output or BASE_DIR / "eval_outputs" / f"rescore_{timestamp}"
        asyncio.run(rescore_saved(args, output))
    else:
        cases = load_cases(args.cases)
        if args.limit:
            cases = cases[:args.limit]
        missing = sum(not (case.get("reference") or "").strip() for case in cases)
        if missing:
            print(f"{missing} question(s) have no reference: precision/recall will be skipped.")
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        output = args.output or BASE_DIR / "eval_outputs" / timestamp
        asyncio.run(run_evaluation(args, cases, output))
    print(f"Open graph report: {(output / 'report.html').resolve()}")


if __name__ == "__main__":
    # Keep the familiar command working from the activated baseline environment.
    # Plot-only/help need no external dependencies and can run in either environment.
    if (EVAL_PYTHON.exists() and Path(sys.executable).resolve() != EVAL_PYTHON.resolve()
            and "--plot-only" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv):
        raise SystemExit(subprocess.call([str(EVAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]))
    main()
