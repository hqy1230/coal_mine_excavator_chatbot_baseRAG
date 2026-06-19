"""Contextual bandit over retrieval depth k (online RL on RAG path)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from utils.path_tools import get_abs_path

from optimization.reward import compute_rag_reward


def _bucket_query(q: str) -> int:
    q = (q or "").strip()
    flags = 0
    for i, kw in enumerate(("报告", "记录", "故障", "卡", "天气", "选购", "拖布", "地图")):
        if kw in q:
            flags |= 1 << i
    return (min(len(q), 120) % 13) * 256 + (flags % 256)


class RetrievalRLPolicy:
    """
    Epsilon-greedy bandit: state = discretized query bucket, action = index into k_candidates.
    Q updated with sample mean (incremental).
    """

    def __init__(
        self,
        k_candidates: tuple[int, ...] = (2, 4, 6),
        epsilon: float = 0.12,
        artifact_path: str = "optimization/artifacts/rl_bandit.json",
    ):
        self.k_candidates = k_candidates
        self.epsilon = epsilon
        self._path = Path(get_abs_path(artifact_path))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.q: dict[str, list[float]] = {}
        self.n: dict[str, list[int]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self.q = {k: list(map(float, v)) for k, v in raw.get("q", {}).items()}
            self.n = {k: list(map(int, v)) for k, v in raw.get("n", {}).items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            self.q, self.n = {}, {}

    def _save(self) -> None:
        payload = {"q": self.q, "n": self.n}
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def state_key(self, query: str) -> str:
        return str(_bucket_query(query))

    def choose_k(self, query: str) -> tuple[int, int]:
        """Returns (k, action_index)."""
        sk = self.state_key(query)
        n_actions = len(self.k_candidates)
        if sk not in self.q:
            self.q[sk] = [0.0] * n_actions
            self.n[sk] = [0] * n_actions

        if random.random() < self.epsilon:
            a = random.randrange(n_actions)
        else:
            a = max(range(n_actions), key=lambda i: self.q[sk][i])
        return self.k_candidates[a], a

    def observe(self, query: str, action_index: int, context: str, reply: str) -> float:
        sk = self.state_key(query)
        r = compute_rag_reward(query, context, reply)
        if sk not in self.q:
            self.q[sk] = [0.0] * len(self.k_candidates)
            self.n[sk] = [0] * len(self.k_candidates)
        n = self.n[sk][action_index] + 1
        old = self.q[sk][action_index]
        self.q[sk][action_index] = old + (r - old) / n
        self.n[sk][action_index] = n
        self._save()
        return r


# Shared policy instance for the process (simple online learning)
rl_retrieval_policy = RetrievalRLPolicy()
