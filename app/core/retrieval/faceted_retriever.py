from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from app.core.caption.schema import CaptionSchema
from app.core.retrieval.reranker_client import RerankerClient
from app.core.retrieval.source_router import mechanism_overlap_score

KB_SOURCE_TYPES = {"chunk", "document", "kb_document"}
CASE_SOURCE_TYPES = {"verified_case", "unverified_case", "case"}


class FacetedRetriever:
    KNOWLEDGE_BOOST = 1.5
    KNOWLEDGE_KEYWORD_BOOST = 2.0

    def __init__(self, reranker: RerankerClient | None = None):
        self.reranker = reranker

    def build_signature(self, caption: CaptionSchema) -> str:
        s = caption.symptoms
        return " ".join(
            [
                f"color:{','.join([x.value for x in s.color])}",
                f"tissue:{','.join([x.value for x in s.tissue_state])}",
                f"shape:{','.join([x.value for x in s.spot_shape])}",
                f"boundary:{','.join([x.value for x in s.boundary])}",
                f"position:{','.join([x.value for x in s.distribution_position])}",
                f"pattern:{','.join([x.value for x in s.distribution_pattern])}",
                f"morph:{','.join([x.value for x in s.morph_change])}",
                f"pest:{','.join([x.value for x in s.pest_cues])}",
                f"cosign:{','.join([x.value for x in s.co_signs])}",
                f"area:{caption.numeric.area_ratio:.3f}",
                f"severity:{caption.numeric.severity_score:.3f}",
            ]
        )

    def _detect_source_type(self, record: dict[str, Any]) -> str:
        entry_type = str(record.get("entry_type", "")).lower()
        if entry_type in KB_SOURCE_TYPES or "chunk" in entry_type:
            return "kb"
        if entry_type in CASE_SOURCE_TYPES or "case" in entry_type:
            return "case"
        if record.get("content") and not record.get("symptoms"):
            return "kb"
        return "case"

    def _extract_text_for_matching(self, record: dict[str, Any], source_type: str) -> str:
        if source_type == "kb":
            parts = [
                record.get("title", ""),
                record.get("content", ""),
                record.get("preview", ""),
            ]
            return " ".join(str(p) for p in parts if p)
        else:
            return json.dumps(record, ensure_ascii=False)

    def _keyword_match_score(self, query: str, text: str) -> float:
        query_lower = query.lower()
        text_lower = text.lower()
        query_keywords = re.findall(r"[\u4e00-\u9fa5]+|[a-z_]+", query_lower)
        if not query_keywords:
            return 0.0
        matched = sum(1 for kw in query_keywords if kw in text_lower)
        return matched / len(query_keywords)

    def _semantic_match_score(self, query: str, text: str) -> float:
        q_clean = re.sub(r"[\s\n]+", " ", query.lower()).strip()
        t_clean = re.sub(r"[\s\n]+", " ", text.lower()).strip()
        if not q_clean or not t_clean:
            return 0.0
        return SequenceMatcher(None, q_clean, t_clean[:1000]).ratio()

    def _compute_combined_score(
        self,
        record: dict[str, Any],
        source_type: str,
        query: str,
        text: str,
        mechanism_tags: dict[str, Any] | None = None,
        source_routing: dict[str, Any] | None = None,
    ) -> float:
        keyword_score = self._keyword_match_score(query, text)
        semantic_score = self._semantic_match_score(query, text)
        mechanism_score = mechanism_overlap_score(mechanism_tags or {}, record.get("mechanism_tags", {}))
        memory_score = 0.0
        try:
            memory_score = max(0.0, min(1.0, float(record.get("memory_score", 0.0) or 0.0)))
        except (TypeError, ValueError):
            memory_score = 0.0
        base_score = keyword_score * 0.45 + semantic_score * 0.35 + mechanism_score * 0.20
        if source_type == "kb":
            base_score *= self.KNOWLEDGE_BOOST
            if keyword_score > 0.3:
                base_score *= self.KNOWLEDGE_KEYWORD_BOOST
            route_weight = float((source_routing or {}).get("document_weight", 1.0) or 1.0)
            return base_score * route_weight

        reference_bonus = 0.12 if str(record.get("memory_level", "")).strip() == "reference_memory" else 0.0
        route_weight = float((source_routing or {}).get("case_weight", 1.0) or 1.0)
        return (base_score + memory_score * 0.25 + reference_bonus) * route_weight

    def retrieve(
        self,
        caption: CaptionSchema,
        candidates: list[dict[str, Any]],
        k: int = 4,
        query_text: str | None = None,
        mechanism_tags: dict[str, Any] | None = None,
        source_routing: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        primary = (query_text or "").strip()
        query = primary if primary else self.build_signature(caption)
        scored: list[tuple[float, dict[str, Any], str]] = []
        for record in candidates:
            source_type = self._detect_source_type(record)
            text = self._extract_text_for_matching(record, source_type)
            score = self._compute_combined_score(
                record,
                source_type,
                query,
                text,
                mechanism_tags=mechanism_tags,
                source_routing=source_routing,
            )
            enriched = dict(record)
            retrieval = dict(enriched.get("retrieval", {})) if isinstance(enriched.get("retrieval"), dict) else {}
            retrieval.update(
                {
                    "faceted_score": round(float(score), 4),
                    "source_type": source_type,
                    "mechanism_overlap_score": round(
                        float(mechanism_overlap_score(mechanism_tags or {}, enriched.get("mechanism_tags", {}))),
                        4,
                    ),
                }
            )
            enriched["retrieval"] = retrieval
            scored.append((score, enriched, text))
        scored.sort(key=lambda x: x[0], reverse=True)

        if self.reranker and scored:
            docs = [item[2] for item in scored]
            rerank_scores = self.reranker.rerank(query=query, documents=docs)
            if rerank_scores is not None and len(rerank_scores) == len(scored):
                reranked: list[tuple[float, dict[str, Any]]] = []
                for idx, (_, record, _) in enumerate(scored):
                    source_type = self._detect_source_type(record)
                    rerank_score = rerank_scores[idx]
                    if source_type == "kb":
                        rerank_score *= self.KNOWLEDGE_BOOST * float((source_routing or {}).get("document_weight", 1.0) or 1.0)
                    else:
                        memory_score = 0.0
                        try:
                            memory_score = max(0.0, min(1.0, float(record.get("memory_score", 0.0) or 0.0)))
                        except (TypeError, ValueError):
                            memory_score = 0.0
                        rerank_score = (
                            rerank_score * float((source_routing or {}).get("case_weight", 1.0) or 1.0)
                            + memory_score * 0.15
                            + mechanism_overlap_score(mechanism_tags or {}, record.get("mechanism_tags", {})) * 0.1
                        )
                    retrieval = dict(record.get("retrieval", {})) if isinstance(record.get("retrieval"), dict) else {}
                    retrieval["rerank_score"] = round(float(rerank_score), 4)
                    record["retrieval"] = retrieval
                    reranked.append((rerank_score, record))
                reranked.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in reranked[:k]]

        return [item[1] for item in scored[:k]]
