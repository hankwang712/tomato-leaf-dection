from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.core.retrieval.kb_chunk_embed import (
    ChunkVectorStore,
    KnowledgeChunkEmbedder,
    Qwen3KnowledgeChunkEmbedder,
    append_chunks_jsonl,
    chunk_text_by_tokens,
    load_chunks_jsonl,
    top_k_cosine,
    truncate_file,
)
from app.core.retrieval.reranker_client import RerankerClient


class GovernanceKnowledgeBase:
    """知识库：病例 JSONL + 文档（分块、向量索引；默认可仅用 embedding Top-K，可选 KB_USE_RERANKER）。"""

    def __init__(
        self,
        kb_dir: str,
        *,
        settings: Settings | None = None,
        reranker: RerankerClient | None = None,
    ):
        self.base_dir = Path(kb_dir)
        self.verified_dir = self.base_dir / "verified"
        self.unverified_dir = self.base_dir / "unverified"
        self.documents_dir = self.base_dir / "documents"
        self.verified_file = self.verified_dir / "cases.jsonl"
        self.unverified_file = self.unverified_dir / "cases.jsonl"
        self.documents_file = self.documents_dir / "documents.jsonl"
        self.chunks_file = self.documents_dir / "chunks.jsonl"
        self.chunk_embeddings_npz = self.documents_dir / "chunk_embeddings.npz"

        self.settings = settings
        self.reranker = reranker
        self._index_lock = Lock()
        self._embedder: KnowledgeChunkEmbedder | Qwen3KnowledgeChunkEmbedder | None = None
        self._vector_store: ChunkVectorStore | None = None

        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.unverified_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def _vector(self) -> ChunkVectorStore:
        if self._vector_store is None:
            self._vector_store = ChunkVectorStore(self.chunk_embeddings_npz)
        return self._vector_store

    @staticmethod
    def _resolve_kb_embedding_backend(settings: Settings) -> str:
        raw = str(settings.kb_embedding_backend or "auto").strip().lower()
        if raw not in {"", "auto"}:
            return raw
        mid = str(settings.kb_embedding_model).lower()
        if "qwen3-embedding" in mid or "qwen3_embedding" in mid:
            return "qwen3"
        return "sentence_transformers"

    def _get_embedder(self) -> KnowledgeChunkEmbedder | Qwen3KnowledgeChunkEmbedder:
        if self.settings is None:
            raise RuntimeError("知识库向量索引需要 Settings（请从 DiagnosisPipeline 注入）。")
        mid = str(self.settings.kb_embedding_model).strip()
        if not mid:
            raise RuntimeError("KB_EMBEDDING_MODEL 未配置。")
        if self._embedder is None:
            dev = str(self.settings.kb_embedding_device or "auto").strip()
            backend = self._resolve_kb_embedding_backend(self.settings)
            if backend == "qwen3":
                self._embedder = Qwen3KnowledgeChunkEmbedder(
                    mid,
                    device=None if dev.lower() in {"", "auto"} else dev,
                    batch_size=self.settings.kb_embedding_batch_size,
                    max_length=self.settings.kb_qwen_max_length,
                    embedding_dim=self.settings.kb_qwen_embedding_dim,
                    instruction=self.settings.kb_qwen_query_instruction or None,
                )
            elif backend == "sentence_transformers":
                self._embedder = KnowledgeChunkEmbedder(
                    mid,
                    device=None if dev.lower() in {"", "auto"} else dev,
                    batch_size=self.settings.kb_embedding_batch_size,
                )
            else:
                raise ValueError(
                    f"KB_EMBEDDING_BACKEND 无效：{backend}，请使用 auto、qwen3 或 sentence_transformers。"
                )
        return self._embedder

    def save_document(
        self,
        *,
        title: str,
        content: str,
        source_name: str = "",
        content_format: str = "text",
    ) -> dict[str, Any]:
        clean_title = str(title).strip() or "未命名知识条目"
        clean_content = str(content).strip()
        if not clean_content:
            raise ValueError("文档内容为空")

        clean_format = str(content_format).strip().lower() or "text"
        if clean_format not in {"text", "md"}:
            clean_format = "text"

        if self.settings is None:
            raise RuntimeError("save_document 向量分块需要 Settings。")

        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        preview = clean_content.replace("\r", " ").replace("\n", " ")[:220]
        ts = datetime.now().isoformat()

        embedder = self._get_embedder()
        tokenizer = embedder.tokenizer
        max_tok = max(64, int(self.settings.kb_chunk_max_tokens))
        overlap = max(0, min(int(self.settings.kb_chunk_overlap_tokens), max_tok - 1))
        pieces = chunk_text_by_tokens(
            clean_content,
            tokenizer,
            max_tokens=max_tok,
            overlap_tokens=overlap,
        )
        if not pieces:
            pieces = [clean_content[:2000]]

        vectors = embedder.encode(pieces, is_query=False)
        chunk_rows: list[dict[str, Any]] = []
        chunk_ids: list[str] = []
        for idx, piece in enumerate(pieces):
            chunk_id = f"{doc_id}_c{idx:04d}"
            chunk_ids.append(chunk_id)
            pv = piece.replace("\r", " ").replace("\n", " ")[:220]
            chunk_rows.append(
                {
                    "entry_type": "chunk",
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "title": clean_title,
                    "source_name": str(source_name).strip(),
                    "content_format": clean_format,
                    "content": piece,
                    "char_count": len(piece),
                    "preview": pv,
                    "timestamp": ts,
                }
            )

        meta = {
            "entry_type": "document_meta",
            "doc_id": doc_id,
            "title": clean_title,
            "source_name": str(source_name).strip(),
            "content_format": clean_format,
            "char_count": len(clean_content),
            "chunk_count": len(pieces),
            "preview": preview,
            "timestamp": ts,
        }

        with self._index_lock:
            append_chunks_jsonl(self.chunks_file, chunk_rows)
            self._vector().append(chunk_ids, vectors)
            with self.documents_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        return meta

    def load_documents(self) -> list[dict[str, Any]]:
        """列出用户可见的文档目录（按 doc_id 去重，优先 document_meta）。"""
        return self.load_document_catalog()

    def load_document_catalog(self) -> list[dict[str, Any]]:
        raw = self._load_jsonl(self.documents_file)
        by_id: dict[str, dict[str, Any]] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            et = str(row.get("entry_type", "document")).strip()
            if et == "document_meta":
                by_id[doc_id] = row
            elif et == "document":
                if doc_id not in by_id:
                    by_id[doc_id] = row
            else:
                continue
        items = list(by_id.values())
        items.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        return items

    def retrieve_documents(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        q = str(query or "").strip()
        if not q:
            return []

        if self.settings is None:
            raise RuntimeError("retrieve_documents 需要 Settings（含 KB_EMBEDDING_* 配置）。")

        use_rerank = bool(self.settings.kb_use_reranker) and self.reranker is not None and self.reranker.is_enabled()
        if bool(self.settings.kb_use_reranker) and not use_rerank:
            raise ValueError(
                "已设置 KB_USE_RERANKER=true，但 Reranker 未就绪，请配置 RERANKER_BASE_URL 与 RERANKER_MODEL。"
            )

        with self._index_lock:
            self._backfill_legacy_documents_locked()

        chunks = load_chunks_jsonl(self.chunks_file)
        chunk_ids, matrix = self._vector().load()
        if not chunks or matrix.size == 0 or len(chunks) != len(chunk_ids):
            if chunks and len(chunks) != len(chunk_ids):
                raise RuntimeError(
                    "chunks.jsonl 与 chunk_embeddings.npz 行数不一致，请清空知识文档后重新上传。"
                )
            return []

        query_vec = self._get_embedder().encode([q], is_query=True)[0]
        if matrix.shape[1] != query_vec.shape[0]:
            raise RuntimeError("当前向量维度与嵌入模型不一致，请清空知识文档后重新索引。")

        cand_k = max(k, int(self.settings.kb_retrieval_candidate_k)) if use_rerank else k
        ranked = top_k_cosine(query_vec, matrix, cand_k)
        if not ranked:
            return []

        if not use_rerank:
            out_emb: list[dict[str, Any]] = []
            for idx, sim in ranked[:k]:
                if not (0 <= idx < len(chunks)):
                    continue
                row = dict(chunks[idx])
                row["retrieval"] = {
                    "stage": "embedding_only",
                    "chunk_row_index": int(idx),
                    "embedding_similarity": float(sim),
                }
                out_emb.append(row)
            return out_emb

        idxs = [i for i, _ in ranked]
        emb_sims = [float(s) for _, s in ranked]
        candidates = [chunks[i] for i in idxs if 0 <= i < len(chunks)]
        if len(candidates) != len(idxs):
            return []
        doc_texts = [f"{c.get('title', '')}\n{c.get('content', '')}" for c in candidates]
        assert self.reranker is not None
        scores = self.reranker.rerank(query=q, documents=doc_texts)
        if scores is None or len(scores) != len(candidates):
            raise RuntimeError("Reranker 调用失败或返回分数长度与候选不一致，请检查 RERANKER_* 接口。")

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out_rr: list[dict[str, Any]] = []
        for ci in order[:k]:
            row = dict(candidates[ci])
            row["retrieval"] = {
                "stage": "embedding_then_rerank",
                "chunk_row_index": int(idxs[ci]),
                "embedding_similarity": emb_sims[ci],
                "rerank_score": float(scores[ci]),
            }
            out_rr.append(row)
        return out_rr

    def _backfill_legacy_documents_locked(self) -> None:
        """将旧版整文件 document 行切分、编码并写入 chunks（幂等：已有 chunk 的 doc_id 跳过）。"""
        if self.settings is None:
            return
        indexed_docs = {str(c.get("doc_id", "")).strip() for c in load_chunks_jsonl(self.chunks_file)}
        raw = self._load_jsonl(self.documents_file)
        for row in raw:
            if not isinstance(row, dict):
                continue
            if str(row.get("entry_type", "")).strip() == "document_meta":
                continue
            if str(row.get("entry_type", "document")).strip() != "document":
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            content = str(row.get("content", "")).strip()
            if not doc_id or not content:
                continue
            if doc_id in indexed_docs:
                continue

            title = str(row.get("title", "")).strip() or "未命名知识条目"
            source_name = str(row.get("source_name", "")).strip()
            content_format = str(row.get("content_format", "text")).strip().lower() or "text"

            embedder = self._get_embedder()
            tokenizer = embedder.tokenizer
            max_tok = max(64, int(self.settings.kb_chunk_max_tokens))
            overlap = max(0, min(int(self.settings.kb_chunk_overlap_tokens), max_tok - 1))
            pieces = chunk_text_by_tokens(content, tokenizer, max_tokens=max_tok, overlap_tokens=overlap)
            if not pieces:
                pieces = [content[:2000]]
            vectors = embedder.encode(pieces, is_query=False)
            ts = str(row.get("timestamp", datetime.now().isoformat()))
            chunk_rows: list[dict[str, Any]] = []
            chunk_ids: list[str] = []
            for idx, piece in enumerate(pieces):
                chunk_id = f"{doc_id}_c{idx:04d}"
                chunk_ids.append(chunk_id)
                pv = piece.replace("\r", " ").replace("\n", " ")[:220]
                chunk_rows.append(
                    {
                        "entry_type": "chunk",
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "title": title,
                        "source_name": source_name,
                        "content_format": content_format,
                        "content": piece,
                        "char_count": len(piece),
                        "preview": pv,
                        "timestamp": ts,
                    }
                )
            append_chunks_jsonl(self.chunks_file, chunk_rows)
            self._vector().append(chunk_ids, vectors)

            meta = {
                "entry_type": "document_meta",
                "doc_id": doc_id,
                "title": title,
                "source_name": source_name,
                "content_format": content_format,
                "char_count": len(content),
                "chunk_count": len(pieces),
                "preview": content.replace("\r", " ").replace("\n", " ")[:220],
                "timestamp": ts,
            }
            with self.documents_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
            indexed_docs.add(doc_id)

    def load_all_knowledge(self) -> dict[str, Any]:
        verified = self.load_cases(verified=True)
        unverified = self.load_cases(verified=False)
        catalog = self.load_document_catalog()
        return {
            "verified": verified,
            "unverified": unverified,
            "documents": catalog,
            "total_verified": len(verified),
            "total_unverified": len(unverified),
            "total_documents": len(catalog),
        }

    def save_case(self, record: dict[str, Any], verified: bool) -> None:
        target = self.verified_file if verified else self.unverified_file
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_cases(self, verified: bool) -> list[dict[str, Any]]:
        source = self.verified_file if verified else self.unverified_file
        return self._load_jsonl(source)

    def retrieve_text(self, query: str, verified: bool = True, k: int = 3) -> list[dict[str, Any]]:
        records = self.load_cases(verified=verified)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            text = json.dumps(record, ensure_ascii=False)
            score = SequenceMatcher(None, query, text).ratio()
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def should_write_verified(self, trace: dict[str, Any], safety: dict[str, Any]) -> bool:
        if not bool(safety.get("safety_passed", False)):
            return False

        rounds = trace.get("rounds", [])
        if not rounds:
            return False

        has_citation = False
        has_supporting_evidence = False
        for round_item in rounds:
            for turn in round_item.get("expert_turns", []):
                agent_name = str(turn.get("agent_name", "")).strip()
                if not agent_name or agent_name == "unknown":
                    continue
                if bool(turn.get("invalid_turn", False)):
                    continue

                citations = [str(x).strip() for x in turn.get("citations", []) if str(x).strip()]
                citations = [x for x in citations if not x.startswith("fallback_")]
                if citations:
                    has_citation = True
                evidence_board = turn.get("evidence_board", [])
                has_board_support = any(
                    isinstance(item, dict) and any(str(part).strip() for part in item.get("supporting", []))
                    for item in evidence_board
                )
                if turn.get("supporting_evidence") or has_board_support:
                    has_supporting_evidence = True
        return has_citation and has_supporting_evidence

    def clear_cases(self, target: str = "all") -> dict[str, int | str]:
        scope = target.strip().lower() if isinstance(target, str) else "all"
        if scope not in {"all", "verified", "unverified", "documents"}:
            raise ValueError("target 仅支持 all（全部）、verified（已核实）、unverified（待核实）、documents（知识文档）")

        before_verified = self._count_cases(self.verified_file)
        before_unverified = self._count_cases(self.unverified_file)
        before_documents = self._count_cases(self.documents_file)
        before_chunks = self._count_cases(self.chunks_file)

        if scope in {"all", "verified"}:
            self._truncate_file(self.verified_file)
        if scope in {"all", "unverified"}:
            self._truncate_file(self.unverified_file)
        if scope in {"all", "documents"}:
            self._truncate_file(self.documents_file)
            truncate_file(self.chunks_file)
            self._vector().clear()
            self._vector_store = ChunkVectorStore(self.chunk_embeddings_npz)

        after_verified = self._count_cases(self.verified_file)
        after_unverified = self._count_cases(self.unverified_file)
        after_documents = self._count_cases(self.documents_file)
        after_chunks = self._count_cases(self.chunks_file)

        return {
            "target": scope,
            "before_verified": before_verified,
            "before_unverified": before_unverified,
            "before_documents": before_documents + before_chunks,
            "after_verified": after_verified,
            "after_unverified": after_unverified,
            "after_documents": after_documents + after_chunks,
            "cleared_total": (
                (before_verified - after_verified)
                + (before_unverified - after_unverified)
                + (before_documents - after_documents)
                + (before_chunks - after_chunks)
            ),
        }

    def _load_jsonl(self, source: Path) -> list[dict[str, Any]]:
        if not source.exists():
            return []
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    @staticmethod
    def _count_cases(source: Path) -> int:
        if not source.exists():
            return 0
        count = 0
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    @staticmethod
    def _truncate_file(source: Path) -> None:
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", encoding="utf-8"):
            pass
