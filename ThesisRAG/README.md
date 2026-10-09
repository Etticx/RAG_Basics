## setup
- python -m venv venv #for virtual env
- .venv/Scripts/Activate.ps1 #activate virtual env
- pip install -r requirements.txt #for installing requirements

## Evaluate retrieval and answers

Metrics run **after answer generation** in `evaluate_rag.py::score_answer()`.
`main.py::build_rag_chain()` retains the retrieved documents beside the answer,
so evaluation uses the same chunks the answer model saw, without retrieving twice.
Importing `main.py` no longer starts indexing or makes API calls.

From this `ThesisRAG` directory, with your virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m venv .eval-venv
.\.eval-venv\Scripts\python.exe -m pip install -r requirements-eval.txt
python evaluate_rag.py --limit 1
```

Ragas 0.4.3 imports a Vertex AI module removed in LangChain Community 0.4.2.
The isolated `.eval-venv` pins Community 0.4.1 for scoring; the original `venv`
continues to load documents, retrieve chunks and generate answers through
`capture_rag.py`. The runner automatically delegates scoring to `.eval-venv`,
so the command above still works from your activated original environment.
Do not install `requirements-eval.txt` into the baseline `venv`.
If your baseline uses a different Python, pass `--rag-python FULL_PATH_TO_PYTHON`.
Evaluation uses your `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
from `.env`. It makes additional Gemini calls to judge answers and compute answer
relevancy embeddings, so quota/cost is higher than asking the chatbot alone.

Edit `eval_cases.json`: replace each `null` reference with an answer you have
checked against the thesis. Add more questions as you learn. Without references,
faithfulness and answer relevancy still run, but context precision and recall are
skipped. Unavailable scores stay blank/null, never zero. The three supplied
questions are starting points, not a validated benchmark.

```json
{"question": "Your thesis question", "reference": "Your verified answer from the thesis"}
```

Run all questions with `python evaluate_rag.py`. Each run saves a timestamped
folder under `eval_outputs/` containing:

- `report.html`: average-score graph, per-question score heatmap, expandable
  answers and the actual retrieved chunks.
- `metric_averages.svg` and `scores_by_question.svg`: standalone graph files.
  Graphs omit metrics with no successful scores.
- `scores.csv`: numeric scores; blanks mean skipped/failed.
- `results.json`: scores, references, answers, source metadata, chunks and notes.
  This is checkpointed after each question.

Open `report.html` in your browser. Graphs use standard-library SVG rendering,
so no Matplotlib dependency is needed. Rebuild graphs from a saved run without
making API calls:

```powershell
python evaluate_rag.py --plot-only eval_outputs/YOUR_RUN/results.json
```

If scoring fails after the baseline generated an answer, retry scoring its saved
answer and exact chunks without reindexing or generating a new answer:

```powershell
python evaluate_rag.py --rescore eval_outputs/YOUR_RUN/results.json
```

This creates a separate report folder and preserves the original run. It still
makes Gemini calls for scoring. The judge uses an asynchronous client because
Ragas collections metrics call `agenerate()`. Errors include redacted diagnostic
messages; successfully rescored metrics have stale failure notes removed.

Judge requests are spaced 13 seconds apart by default for the observed five
requests/minute Gemini quota. HTTP 429 and transient server errors get up to two
retries, honoring Google's retry delay when supplied. Set
`RAG_EVAL_REQUEST_INTERVAL` in `.env` if your quota needs a different interval.
This changes evaluation pacing only, not questions, answers or metric definitions.

The answer model defaults to the model already used by this project. Ensure that
model is available to your API key; model availability has not been tested here.
Override models without editing code:

```powershell
python evaluate_rag.py --model YOUR_GEMINI_MODEL --judge-model YOUR_JUDGE_MODEL
```

Alternatively set `RAG_MODEL`, `RAG_EVAL_MODEL` and optionally
`RAG_EVAL_EMBEDDING_MODEL` in `.env`. Evaluation embeddings default to
`gemini-embedding-001`, separate from the local retrieval embeddings. Using the
same model as generator and judge can introduce correlated errors; manually
review a sample of answers. Faithfulness checks support in retrieved text,
not truth of the thesis, and relevancy does not establish factual correctness.

This evaluation retains the original prompt, chunking and top-k settings so you
can measure them before tuning. The existing persistent indexing behavior is
also retained: repeated indexing may add duplicates. Outputs contain thesis text
and are ignored by Git through `eval_outputs/`.

Implementation follows Ragas' [collections metrics API](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
and its documented [Gemini integration](https://docs.ragas.io/en/latest/howtos/integrations/gemini/).
