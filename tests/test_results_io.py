import json

import pytest

from utils.results_io import (export_report, load_results, markdown_to_html, save_results)

R = {"stroop": {"accuracy": 0.9, "n": 40, "rt": [1, 2]}, "flanker": {"accuracy": 0.8, "n": 30}}


def test_csv_roundtrip_is_really_csv(tmp_path):
    p = tmp_path / "r.csv"
    assert save_results(p, R) == "csv"
    text = p.read_text()
    assert text.startswith("experiment,") and not text.lstrip().startswith("{")
    assert load_results(p)["stroop"]["accuracy"] == 0.9 and load_results(p)["stroop"]["rt"] == [1, 2]


def test_json_roundtrip_and_validation(tmp_path):
    p = tmp_path / "r.json"
    save_results(p, R)
    assert load_results(p) == R
    p.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError):
        load_results(p)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_results(bad)
    with pytest.raises(ValueError):
        save_results(tmp_path / "x.xlsx", R)


def test_report_formats_match_extension(tmp_path):
    md = "# Title\n\n**Total:** 3 <script>x</script>\n- item `a`\n"
    h = tmp_path / "r.html"
    assert export_report(h, md) == "html"
    t = h.read_text()
    assert t.startswith("<!doctype html>") and "<h1>Title</h1>" in t and "<script>" not in t and "&lt;script&gt;" in t
    f = tmp_path / "r.pdf"
    assert export_report(f, md) == "pdf" and f.read_bytes().startswith(b"%PDF")
    m = tmp_path / "r.md"
    assert export_report(m, md) == "markdown" and m.read_text() == md
    assert "<ul>" in markdown_to_html(md)
