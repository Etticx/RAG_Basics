"""Capture baseline answers/evidence in its original environment, without Ragas."""

import argparse
import json
from pathlib import Path

from main import answer_question, build_rag_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8-sig"))
    if args.limit:
        cases = cases[:args.limit]
    chain = build_rag_chain(model=args.model)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] Baseline RAG: {case['question']}", flush=True)
        try:
            row = answer_question(chain, case["question"])
            row.update(reference=case.get("reference"), notes={})
        except Exception as error:
            row = dict(question=case["question"], reference=case.get("reference"),
                       answer=None, contexts=[], sources=[],
                       notes={"generation": f"Failed: {type(error).__name__}."})
            print(f"Generation failed ({type(error).__name__}).", flush=True)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
