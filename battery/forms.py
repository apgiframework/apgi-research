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
