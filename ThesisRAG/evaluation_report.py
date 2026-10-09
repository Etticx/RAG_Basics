"""Render saved evaluation scores as SVG graphs and an HTML evidence report.

Uses only Python's standard library; graph rendering makes no API calls.
"""

import html
import math
from pathlib import Path
from statistics import mean

METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}


def escape(value):
    return html.escape(str(value), quote=True)


def valid_score(row, name):
    value = row.get(name)
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (-1 if name == "answer_relevancy" else 0) <= value <= 1)


def text(x, y, value, size=14, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="#243447">{escape(value)}</text>')


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f'<title>{escape(title)}</title>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<g font-family="Arial, sans-serif">', text(24, 32, title, 20)]


def build_graphs(payload):
    rows = payload["rows"]
    active = [name for name in METRICS if any(valid_score(row, name) for row in rows)]
    if not active:
        return {}
    low = -1 if any(row[name] < 0 for row in rows for name in active
                    if valid_score(row, name)) else 0
    caption = f"Source: Ragas evaluator | Run: {payload.get('created_at', 'unspecified')}"
    # Horizontal bars: average only successful scores; show each denominator.
    width, height = 1000, 135 + len(active) * 55
    parts = svg_start(width, height, "Mean RAG evaluation scores")
    parts.append(text(24, 57, "Means exclude skipped/failed scores; n is the successful question count.", 13))
    x0, plot_width = 225, 560
    axis_y = 80 + len(active) * 55
    position = lambda value: x0 + (value - low) / (1 - low) * plot_width
    for tick in ([0, .25, .5, .75, 1] if low == 0 else [-1, -.5, 0, .5, 1]):
        x = position(tick)
        parts.append(f'<line x1="{x}" y1="72" x2="{x}" y2="{axis_y}" stroke="#e2e8f0"/>')
        parts.append(text(x, axis_y + 18, f"{tick:g}", 12, "middle"))
    for i, name in enumerate(active):
        values = [row[name] for row in rows if valid_score(row, name)]
        score, y = mean(values), 83 + i * 55
        parts.append(text(24, y + 20, METRICS[name]))
        start, end = sorted((position(0), position(score)))
        parts.append(f'<rect x="{start}" y="{y}" width="{end-start}" height="28" fill="#2563a6"/>')
        parts.append(text(805, y + 20, f"{score:.3f} (n={len(values)}/{len(rows)})"))
    parts.extend([text(505, axis_y + 40, "Evaluator score (unitless; higher is better)", 13, "middle"),
                  text(24, height - 10, caption, 11), '</g></svg>'])
    average_svg = "\n".join(parts)

    # Heatmap: exact scores for every question, retaining missing values as dashes.
    width, height = max(720, 235 + len(rows) * 82), 150 + len(active) * 50
    parts = svg_start(width, height, "RAG evaluation scores by question")
    parts.append(text(24, 56, "Cell values are evaluator scores; a dash means skipped or failed.", 13))
    for i, row in enumerate(rows):
        x = 220 + i * 82
        parts.append(text(x + 37, 84, f"Q{i+1}", 13, "middle"))
        for j, name in enumerate(active):
            y = 98 + j * 50
            valid = valid_score(row, name)
            value = row[name] if valid else None
            opacity = .12 + .6 * (value-low)/(1-low) if valid else 1
            color = "#2563a6" if valid else "#edf0f3"
            parts.append(f'<rect x="{x}" y="{y}" width="74" height="40" '
                         f'fill="{color}" fill-opacity="{opacity}"/>')
            parts.append(text(x+37, y+25, f"{value:.2f}" if valid else "—", 14, "middle"))
    for j, name in enumerate(active):
        parts.append(text(24, 123 + j * 50, METRICS[name]))
    y = 105 + len(active) * 50
    parts.append(text(220, y, "Question number (see evidence below)", 13))
    parts.append(text(24, y + 21, f"Color scale: {low} to 1, light to dark. Scores are unitless; higher is better.", 12))
    parts.extend([text(24, height - 8, caption, 11), '</g></svg>'])
    return {"metric_averages.svg": average_svg, "scores_by_question.svg": "\n".join(parts)}


def write_report(payload, output):
    if not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise ValueError("Saved results must contain a non-empty rows list.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graphs = build_graphs(payload)
    sections = []
    for i, row in enumerate(payload["rows"], start=1):
        scores = "; ".join(f"{label}: {row[name]:.3f}" if valid_score(row, name)
                           else f"{label}: unavailable" for name, label in METRICS.items())
        contexts = "".join(f"<li><pre>{escape(context)}</pre></li>" for context in row.get("contexts", []))
        notes = "".join(f"<li>{escape(name)}: {escape(note)}</li>" for name, note in row.get("notes", {}).items())
        sections.append(f"""<details><summary>Q{i}: {escape(row['question'])}</summary>
<p>{escape(scores)}</p><h3>Generated answer</h3><pre>{escape(row.get('answer') or 'Generation failed.')}</pre>
<h3>Verified reference</h3><pre>{escape(row.get('reference') or 'Not provided.')}</pre>
<h3>Chunks supplied to the answer model</h3><ol>{contexts}</ol><ul>{notes}</ul></details>""")
    for filename, svg in graphs.items():
        (output / filename).write_text(svg, encoding="utf-8")
    metadata = f"Answer model: {payload.get('model', 'unspecified')} | Judge: {payload.get('judge_model', 'unspecified')} | Run: {payload.get('created_at', 'unspecified')}"
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Thesis RAG evaluation</title>
<style>body{font:16px/1.5 Arial,sans-serif;color:#243447;max-width:1100px;margin:30px auto;padding:0 20px}
h1{font-size:26px}h3{font-size:16px}small{color:#526274}.graph{overflow-x:auto;margin:24px 0}
details{border-top:1px solid #d9e0e7;padding:16px 0}summary{cursor:pointer;font-weight:bold}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;font:14px/1.6 monospace}</style>
<h1>Thesis RAG evaluation</h1>"""
    content += f"<small>{escape(metadata)}</small>"
    content += "<p>These are automated judge scores, not probabilities that an answer is correct. Reference answers must be checked against the thesis. Missing/failed scores are excluded from averages.</p>"
    if not graphs:
        content += "<p>No metric succeeded, so no score graphs were generated. Check the failure notes below.</p>"
    for svg in graphs.values():
        content += f'<div class="graph">{svg}</div>'
    content += "<h2>Answers and retrieval evidence</h2>" + "\n".join(sections) + "</html>"
    (output / "report.html").write_text(content, encoding="utf-8")
