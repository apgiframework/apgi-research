"""Content and structure linter for the item bank. Pure functions, no I/O
except `load_bank`. Every rule returns human-readable error strings."""

import json
import pathlib
import re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
AGREEMENT_5 = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]
RESPONSE_TYPE = "agreement_5"
ITEM_KEYS = {
    "item_id",
    "block",
    "subscale",
    "text",
    "response_type",
    "options",
    "source",
}
REQUIRED_KEYS = ITEM_KEYS - {"source"}

# S1-L1-15: exclusion rests on diagnostic function, not licence.
DIAGNOSTIC_INSTRUMENTS = json.loads(
    (ROOT / "exclusion_record.json").read_text(encoding="utf-8")
)["excluded_instruments"]

# S1-L1-16: banned topics. Stems, matched case-insensitively on word starts.
BANNED_CONTENT = {
    "suicidality": r"\bsuicid\w*|\bkill (?:myself|yourself)\b|\bend (?:my|your) (?:own )?life\b|\bwish i (?:were|was) dead\b",
    "self-harm": r"\bself[- ]?harm\w*|\bself[- ]?injur\w*|\bcut(?:ting)? (?:myself|yourself)\b",
    "disordered eating": r"\banorex\w*|\bbulim\w*|\bbing(?:e|ing) eat\w*|\bpurg(?:e|ing)\b|\bdisordered eating\b",
    "substance use": r"\balcohol\b|\bdrunk\b|\bcocaine\b|\bheroin\b|\bcannabis\b|\bmarijuana\b|\bopioid\w*|\billegal drugs?\b|\bsubstance (?:use|abuse)\b",
    "psychiatric medication": r"\bantidepressant\w*|\bantipsychotic\w*|\bssri\b|\bbenzodiazepine\w*|\bprozac\b|\bxanax\b|\bpsychiatric medication\b|\blithium\b",
}

# S1-L1-10.1: printed parameter labels such as 'Low ε' / 'High ε' on options.
PARAMETER_LABEL = re.compile(
    r"\b(?:low|high|weak|strong)\s*[εθΠπαβγκητρσφ]\b|\b[εθΠπαβγκητρσφ]\s*[=:]\s*(?:low|high)\b",
    re.IGNORECASE,
)


def load_bank(path: pathlib.Path | str | None = None) -> dict[str, Any]:
    target = pathlib.Path(path) if path else ROOT / "candidates.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _word_hits(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        name for name, pat in patterns.items() if re.search(pat, text, re.IGNORECASE)
    ]


def lint_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ident = item.get("item_id", "<missing item_id>")
    missing = REQUIRED_KEYS - item.keys()
    if missing:
        errors.append(f"{ident}: missing keys {sorted(missing)}")
    extra = item.keys() - ITEM_KEYS
    if extra:
        # Catches free-text / slider fields (S1-L1-05, S1-L1-16) and anything unregistered.
        errors.append(f"{ident}: unregistered keys {sorted(extra)}")
    if item.get("response_type") != RESPONSE_TYPE:
        errors.append(
            f"{ident}: response_type must be '{RESPONSE_TYPE}' (no sliders or free text)"
        )
    if item.get("options") != AGREEMENT_5:
        errors.append(
            f"{ident}: options must be exactly the fixed 5-point agreement labels"
        )
    text = str(item.get("text", ""))
    if not text.strip():
        errors.append(f"{ident}: empty text")
    for field in ("text", "options"):
        blob = " ".join(
            map(
                str,
                item.get(field, []) if field == "options" else [item.get(field, "")],
            )
        )
        if PARAMETER_LABEL.search(blob):
            errors.append(f"{ident}: printed parameter label in {field}")
    for topic in _word_hits(text, BANNED_CONTENT):
        errors.append(f"{ident}: banned sensitive content ({topic})")
    for inst in DIAGNOSTIC_INSTRUMENTS:
        names = [inst["name"], *inst.get("aliases", [])]
        hay = f"{text} {item.get('source', '')}".lower()
        if any(re.search(rf"\b{re.escape(n.lower())}\b", hay) for n in names):
            errors.append(f"{ident}: derived from diagnostic instrument {inst['name']}")
    return errors


def lint_bank(bank: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    items = bank.get("items", [])
    order = bank.get("block_order", [])
    if len(set(order)) != len(order):
        errors.append("block_order contains duplicates")
    ids = [i.get("item_id") for i in items]
    if len(set(ids)) != len(ids):
        errors.append("duplicate item_id")
    blocks = {i.get("block") for i in items}
    if items and set(order) != blocks:
        errors.append(
            f"block_order {order} does not match item blocks {sorted(map(str, blocks))}"
        )
    for item in items:
        errors.extend(lint_item(item))
    return errors
