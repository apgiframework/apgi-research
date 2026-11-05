"""Matched parallel forms and order assignment (S3-BAT-03, S3-BAT-04)."""

from typing import Any, Sequence


def split_matched_forms(
    stimuli: Sequence[dict[str, Any]], n_forms: int = 3, key: str = "difficulty"
) -> list[list[dict[str, Any]]]:
    """Deal stimuli into `n_forms` forms of equal size matched on `key`.

    Sorting by difficulty and dealing in a snake order (0,1,2,2,1,0,...) keeps
    form means within one rank-step of each other. Deterministic.
    """
    if n_forms < 2:
        raise ValueError("need at least two forms")
    if len(stimuli) % n_forms:
        raise ValueError(
            f"{len(stimuli)} stimuli do not split into {n_forms} equal forms"
        )
    ordered = sorted(stimuli, key=lambda s: (s[key], str(s.get("id", ""))))
    forms: list[list[dict[str, Any]]] = [[] for _ in range(n_forms)]
    for i, stim in enumerate(ordered):
        lap, pos = divmod(i, n_forms)
        forms[pos if lap % 2 == 0 else n_forms - 1 - pos].append(stim)
    return forms


def balance_report(
    forms: Sequence[Sequence[dict[str, Any]]], key: str = "difficulty"
) -> dict[str, Any]:
    sizes = [len(f) for f in forms]
    means = [sum(s[key] for s in f) / len(f) for f in forms]
    return {
        "sizes": sizes,
        "means": means,
        "max_mean_difference": max(means) - min(means),
    }


def form_for_session(order: Sequence[str], session_index: int) -> str:
    """The form a client must serve at `session_index` (1-based), rotating through
    the order assigned at enrolment."""
    if session_index < 1:
        raise ValueError("session_index is 1-based")
    if not order:
        raise ValueError("empty form order")
    return order[(session_index - 1) % len(order)]


def williams_orders(n_forms: int = 3) -> list[tuple[int, ...]]:
    """Williams-design form orders (S3-BAT-04): every form appears in every
    position equally often and follows every other form equally often. For an
    odd number of forms this needs 2*n sequences (the rows and their reversals)."""
    if n_forms < 2:
        raise ValueError("need at least two forms")
    first = [0]
    lo, hi = 1, n_forms - 1
    while len(first) < n_forms:
        first.append(hi)
        hi -= 1
        if len(first) < n_forms:
            first.append(lo)
            lo += 1
    rows = [tuple((x + i) % n_forms for x in first) for i in range(n_forms)]
    if n_forms % 2:
        rows += [tuple(reversed(r)) for r in rows]
    return rows


def assign_order_at_enrolment(
    enrolment_index: int, form_labels: Sequence[str] = ("A", "B", "C")
) -> list[str]:
    """Order for the participant enrolling at 0-based `enrolment_index`, cycling
    through the Williams design so any block of 2n enrolments is perfectly
    balanced. The result is stored with the participant and never recomputed."""
    if enrolment_index < 0:
        raise ValueError("enrolment_index must be non-negative")
    rows = williams_orders(len(form_labels))
    row = rows[enrolment_index % len(rows)]
    return [form_labels[i] for i in row]
