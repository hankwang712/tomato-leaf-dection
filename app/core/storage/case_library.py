from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.retrieval.source_router import mechanism_overlap_score


class CaseLibrary:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.verified_dir = self.base_dir / "verified"
        self.unverified_dir = self.base_dir / "unverified"
        self.verified_file = self.verified_dir / "cases.jsonl"
        self.unverified_file = self.unverified_dir / "cases.jsonl"
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.unverified_dir.mkdir(parents=True, exist_ok=True)

    def save_case(self, record: dict[str, Any], verified: bool) -> None:
        target = self.verified_file if verified else self.unverified_file
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_cases(self, verified: bool) -> list[dict[str, Any]]:
        source = self.verified_file if verified else self.unverified_file
        return self._load_jsonl(source)

    def load_all_cases(self) -> dict[str, Any]:
        verified = self.load_cases(verified=True)
        unverified = self.load_cases(verified=False)
        return {
            "verified": verified,
            "unverified": unverified,
            "total_verified": len(verified),
            "total_unverified": len(unverified),
        }

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False)

    @staticmethod
    def _memory_score(record: dict[str, Any]) -> float:
        try:
            return max(0.0, min(1.0, float(record.get("memory_score", 0.0) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _preliminary_score(
        cls,
        query: str,
        record: dict[str, Any],
        mechanism_tags: dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        text = cls._record_text(record)
        text_score = SequenceMatcher(None, query, text).ratio()
        mechanism_score = mechanism_overlap_score(mechanism_tags or {}, record.get("mechanism_tags", {}))
        memory_score = cls._memory_score(record)
        reference_bonus = 0.12 if str(record.get("memory_level", "")).strip() == "reference_memory" else 0.0
        score = text_score * 0.58 + mechanism_score * 0.22 + memory_score * 0.20 + reference_bonus
        return max(0.0, min(1.0, score)), text_score, mechanism_score

    def estimate_case_support(self, query: str, mechanism_tags: dict[str, Any] | None = None) -> float:
        best = 0.0
        for verified in (True, False):
            for record in self.load_cases(verified=verified):
                score, _, _ = self._preliminary_score(query, record, mechanism_tags)
                best = max(best, score)
        return max(0.0, min(1.0, best))

    def retrieve_text(
        self,
        query: str,
        verified: bool = True,
        k: int = 3,
        mechanism_tags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records = self.load_cases(verified=verified)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            score, text_score, mechanism_score = self._preliminary_score(query, record, mechanism_tags)
            enriched = dict(record)
            enriched["entry_type"] = "verified_case" if verified else "unverified_case"
            enriched["retrieval"] = {
                "stage": "case_memory_prefilter",
                "preliminary_score": round(float(score), 4),
                "text_similarity": round(float(text_score), 4),
                "mechanism_overlap_score": round(float(mechanism_score), 4),
                "memory_score": round(float(self._memory_score(record)), 4),
            }
            scored.append((score, enriched))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def evaluate_case_quality(self, trace: dict[str, Any], safety: dict[str, Any]) -> dict[str, Any]:
        if not bool(safety.get("safety_passed", False)):
            return {
                "write_verified": False,
                "has_citation": False,
                "has_supporting_evidence": False,
            }

        rounds = trace.get("rounds", [])
        if not rounds:
            return {
                "write_verified": False,
                "has_citation": False,
                "has_supporting_evidence": False,
            }

        has_citation = False
        has_supporting_evidence = False
        for round_item in rounds:
            for turn in round_item.get("expert_turns", []):
                agent_name = str(turn.get("agent_name", "")).strip()
                if not agent_name or agent_name == "unknown":
                    continue
                if bool(turn.get("invalid_turn", False)):
                    continue

                citations = [str(value).strip() for value in turn.get("citations", []) if str(value).strip()]
                citations = [value for value in citations if not value.startswith("fallback_")]
                if citations:
                    has_citation = True

                evidence_board = turn.get("evidence_board", [])
                has_board_support = any(
                    isinstance(item, dict) and any(str(part).strip() for part in item.get("supporting", []))
                    for item in evidence_board
                )
                if turn.get("supporting_evidence") or has_board_support:
                    has_supporting_evidence = True

        return {
            "write_verified": has_citation and has_supporting_evidence,
            "has_citation": has_citation,
            "has_supporting_evidence": has_supporting_evidence,
        }

    def should_write_verified(self, trace: dict[str, Any], safety: dict[str, Any]) -> bool:
        return bool(self.evaluate_case_quality(trace, safety).get("write_verified", False))

    def delete_by_run_id(self, run_id: str) -> dict[str, int]:
        removed_verified = self._delete_from_file(self.verified_file, run_id)
        removed_unverified = self._delete_from_file(self.unverified_file, run_id)
        return {
            "removed_verified": removed_verified,
            "removed_unverified": removed_unverified,
            "removed_total": removed_verified + removed_unverified,
        }

    def _delete_from_file(self, source: Path, run_id: str) -> int:
        records = self._load_jsonl(source)
        kept: list[dict[str, Any]] = []
        removed = 0
        for item in records:
            if str(item.get("run_id", "")).strip() == run_id:
                removed += 1
                continue
            kept.append(item)
        if removed:
            self._write_jsonl(source, kept)
        return removed

    def _load_jsonl(self, source: Path) -> list[dict[str, Any]]:
        if not source.exists():
            return []
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    @staticmethod
    def _write_jsonl(source: Path, records: list[dict[str, Any]]) -> None:
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
