"""Deterministic presentation order: fixed block order, items shuffled within
each block with a seed derived from participant/session (S1-L1-06)."""

import hashlib
import random
from typing import Any


def _rng(seed: str, block: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{block}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def render_order(bank: dict[str, Any], seed: str) -> list[str]:
    """Return item_ids in presentation order for `seed` (e.g. session id)."""
    by_block: dict[str, list[str]] = {}
    for item in bank["items"]:
        by_block.setdefault(item["block"], []).append(item["item_id"])
    ordered: list[str] = []
    for block in bank["block_order"]:
        ids = sorted(
            by_block.get(block, [])
        )  # canonical start so the seed alone decides
        _rng(seed, block).shuffle(ids)
        ordered.extend(ids)
    return ordered
