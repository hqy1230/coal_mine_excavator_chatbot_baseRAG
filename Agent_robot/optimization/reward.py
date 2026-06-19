"""Rule-based reward for RL on RAG answers (proxy for human feedback)."""

from __future__ import annotations

import re
from typing import Iterable


def _token_set(text: str) -> set[str]:
    parts = re.findall(r"[\u4e00-\u9fff]{1,2}|[a-zA-Z0-9]+", text or "")
    return set(p.lower() for p in parts if len(p) >= 1)


def overlap_score(query: str, context: str, reply: str) -> float:
    """Token overlap between reply and (context + query), in [0,1]."""
    ref = _token_set(context) | _token_set(query)
    r = _token_set(reply)
    if not r:
        return 0.0
    inter = len(r & ref)
    return inter / max(len(r), 1)


def compute_rag_reward(query: str, context: str, reply: str) -> float:
    """
    Higher is better. Used as RL signal after one RAG completion.
    Not a learned reward model — lightweight heuristics for demo / iteration.
    """
    if not reply or not reply.strip():
        return -2.0

    score = 0.0
    L = len(reply)
    if 40 <= L <= 4000:
        score += 0.4
    elif L < 40:
        score -= 0.3
    else:
        score += 0.1

    score += 1.2 * overlap_score(query, context, reply)

    if "我不知道" in reply and overlap_score(query, context, reply) < 0.05:
        score -= 0.4

    polite = any(p in reply for p in ("建议", "您可以", "谢谢", "抱歉"))
    if polite:
        score += 0.15

    return float(score)
