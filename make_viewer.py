import json
import html
from pathlib import Path

RUNS_DIR = Path("runs")
OUT_HTML = Path("viewer.html")

def load_runs():
    runs = {}
    for run_dir in sorted(RUNS_DIR.glob("v*/")):
        pred_path = run_dir / "run" / "predictions.jsonl"
        if not pred_path.exists():
            continue
        with open(pred_path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        runs[run_dir.name] = rows
    return runs

def main():
    runs = load_runs()
    if not runs:
        print("no runs with predictions.jsonl found")
        return

    run_ids = list(runs.keys())
    first_id = run_ids[0]
    n = len(runs[first_id])
    print(f"loaded {len(runs)} runs: {run_ids}, {n} samples each")

    samples = []
    for i in range(n):
        q = runs[first_id][i]["query"]
        ref = runs[first_id][i]["reference"]
        preds = {rid: (runs[rid][i]["prediction"], runs[rid][i]["rougeL_f"]) for rid in run_ids if i < len(runs[rid])}
        samples.append({"i": i, "q": q, "ref": ref, "preds": preds})

    css = """
    body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 1400px; margin: 1em auto; padding: 0 1em; color:
    .controls { position: sticky; top: 0; background: white; padding: 0.5em 0; border-bottom: 1px solid
    .sample { border: 1px solid
    .num { color:
    .q { background:
    .cols { display: grid; gap: 0.7em; grid-template-columns: 1fr 1fr; }
    .col { background:
    .col.ref { background:
    .col.pred { background:
    .col h4 { margin: 0 0 0.3em 0; font-size: 0.85em; color:
    .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 0.5em; }
    .badge.high { background:
    .badge.mid { background:
    .badge.low { background:
    .body { white-space: pre-wrap; font-size: 0.92em; line-height: 1.4; max-height: 400px; overflow-y: auto; }
    input[type="text"] { padding: 4px 8px; width: 300px; }
    select { padding: 4px; }
    function applyFilter() {
        const q = document.getElementById('q').value.toLowerCase();
        const minR = parseFloat(document.getElementById('minR').value) || 0;
        const maxR = parseFloat(document.getElementById('maxR').value) || 1;
        for (const el of document.querySelectorAll('.sample')) {
            const r = parseFloat(el.dataset.rouge);
            const text = el.dataset.q;
            const ok = text.includes(q) && r >= minR && r <= maxR;
            el.style.display = ok ? '' : 'none';
        }
    }
<title>Predictions viewer</title>
<style>{css}</style></head><body>
<h1>Predictions viewer</h1>
<div class="controls">
  <input id="q" type="text" placeholder="filter by query substring" oninput="applyFilter()">
  ROUGE-L: <input id="minR" type="text" placeholder="min" size="4" oninput="applyFilter()">
  - <input id="maxR" type="text" placeholder="max" size="4" oninput="applyFilter()">
  <span style="margin-left:1em; color:#666;">runs: {", ".join(run_ids)} · samples: {n}</span>
</div>
{"".join(rows_html)}
<script>{js}</script>
</body></html>"""

    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"wrote {OUT_HTML} ({len(html_out)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
