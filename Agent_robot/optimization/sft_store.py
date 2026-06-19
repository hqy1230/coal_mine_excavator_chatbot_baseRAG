"""Load JSONL SFT pairs and retrieve nearest instructions for prompt injection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils.logger_handler import logger
from utils.path_tools import get_abs_path


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class SFTExemplarStore:
    def __init__(self, jsonl_path: str = "data/sft_robot_cs.jsonl", top_n: int = 2):
        self._path = Path(get_abs_path(jsonl_path))
        self.top_n = top_n
        self._rows: list[dict[str, Any]] = []
        self._emb: list[list[float]] | None = None
        self._embed_model = None

    def _ensure_loaded(self) -> None:
        if self._rows:
            return
        if not self._path.exists():
            logger.warning(f"[SFT] 数据文件不存在，跳过 exemplar：{self._path}")
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _ensure_embeddings(self) -> None:
        self._ensure_loaded()
        if not self._rows or self._emb is not None:
            return
        try:
            from model.factory import embed_model

            self._embed_model = embed_model
            texts = [f"{r.get('instruction','')}\n{r.get('input','')}".strip() for r in self._rows]
            self._emb = self._embed_model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"[SFT] 嵌入失败，将不使用 exemplar：{e}")
            self._emb = []

    def format_exemplar_block(self, query: str) -> str:
        self._ensure_embeddings()
        if not self._rows or not self._emb:
            return "（当前无可用监督样例。）"

        try:
            from model.factory import embed_model

            qv = embed_model.embed_query(query)
        except Exception as e:
            logger.warning(f"[SFT] query 嵌入失败：{e}")
            return "（监督样例检索跳过。）"

        scored: list[tuple[float, int]] = []
        for i, ev in enumerate(self._emb):
            scored.append((_cosine(qv, ev), i))
        scored.sort(reverse=True)
        picked = scored[: self.top_n]
        blocks: list[str] = []
        for rank, (_, idx) in enumerate(picked, start=1):
            r = self._rows[idx]
            ins = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            out = (r.get("output") or "").strip()
            user_turn = ins if not inp else f"{ins}\n用户补充：{inp}"
            blocks.append(f"样例{rank}（用户）→\n{user_turn}\n样例{rank}（客服）→\n{out}")
        return "\n\n".join(blocks)


sft_store = SFTExemplarStore()
