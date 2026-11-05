"""Format-honest result import/export for the desktop GUI (AUDIT-GUI-001).

A file's extension must match its content: CSV is written as CSV, HTML as HTML
and PDF as PDF, and an import actually loads (and validates) the data instead of
acknowledging it.
"""

import csv
import html
import io
import json
import re
from pathlib import Path
from typing import Any, Dict


def save_results(path: Path | str, results: Dict[str, dict]) -> str:
    """Write results as JSON or CSV by extension; return the format written."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        keys = sorted({k for r in results.values() for k in r})
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["experiment", *keys])
            for name, r in results.items():
                w.writerow([name, *[json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k, "") for k in keys]])
        return "csv"
    if ext in (".json", ""):
        p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return "json"
    raise ValueError(f"unsupported results format: {ext}")


def _coerce(v: str) -> Any:
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    if v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            pass
    return v


def load_results(path: Path | str) -> Dict[str, dict]:
    """Load results from JSON or CSV; raise ValueError on anything malformed."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "experiment" not in rows[0]:
            raise ValueError("CSV must have an 'experiment' column and at least one row")
        return {r["experiment"]: {k: _coerce(v) for k, v in r.items() if k != "experiment" and v != ""} for r in rows}
    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
            raise ValueError("JSON results must be an object mapping experiment name to an object")
        return data
    raise ValueError(f"unsupported results format: {ext}")


def markdown_to_html(md: str) -> str:
    """Small, escaping Markdown subset (headings, bullets, bold, code)."""
    out, in_list = [], False
    for line in md.splitlines():
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = re.sub(r"`(.+?)`", r"<code>\1</code>", esc)
        m = re.match(r"(#{1,6})\s+(.*)", esc)
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{esc[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>" if m else (f"<p>{esc}</p>" if esc.strip() else ""))
    if in_list:
        out.append("</ul>")
    return "<!doctype html><html><head><meta charset='utf-8'><title>APGI Research Report</title></head><body>\n" + "\n".join(out) + "\n</body></html>\n"


def markdown_to_pdf_bytes(md: str, lines_per_page: int = 52) -> bytes:
    """Render monospaced text pages to a real PDF with matplotlib (a dependency)."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    lines = [w for line in md.splitlines() for w in (textwrap.wrap(line, 95) or [""])]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for i in range(0, max(len(lines), 1), lines_per_page):
            fig = Figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, "\n".join(lines[i:i + lines_per_page]), family="monospace", fontsize=8, va="top")
            pdf.savefig(fig)
    return buf.getvalue()


def export_report(path: Path | str, markdown: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".html":
        p.write_text(markdown_to_html(markdown), encoding="utf-8")
        return "html"
    if ext == ".pdf":
        p.write_bytes(markdown_to_pdf_bytes(markdown))
        return "pdf"
    p.write_text(markdown, encoding="utf-8")
    return "markdown"
