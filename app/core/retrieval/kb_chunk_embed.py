from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


def chunk_text_by_tokens(
    text: str,
    tokenizer: Any,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """按 tokenizer 词元窗口切分，overlap 为相邻块重叠词元数。"""
    clean = str(text or "").strip()
    if not clean:
        return []

    ids = tokenizer.encode(clean, add_special_tokens=False)
    if not ids:
        return [clean]

    overlap = max(0, min(overlap_tokens, max_tokens - 1)) if max_tokens > 1 else 0
    if len(ids) <= max_tokens:
        return [clean]

    chunks: list[str] = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + max_tokens, n)
        piece = ids[start:end]
        decoded = tokenizer.decode(piece, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= n:
            break
    return chunks if chunks else [clean]


class KnowledgeChunkEmbedder:
    """sentence-transformers 编码（L2 归一化）；is_query 参数忽略。"""

    def __init__(self, model_id: str, *, device: str | None = None, batch_size: int = 16):
        self.model_id = str(model_id).strip()
        self.device = (device or "auto").strip().lower()
        self.batch_size = max(1, int(batch_size))
        self._lock = Lock()
        self._model: Any = None

    def _ensure_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.device and self.device not in {"auto", ""}:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_id, **kwargs)
            return self._model

    @property
    def tokenizer(self) -> Any:
        return self._ensure_model().tokenizer

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        _ = is_query
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class Qwen3KnowledgeChunkEmbedder:
    """Qwen3-Embedding 系列（如 Qwen3-Embedding-4B）：查询带 Instruct 前缀，文档块为纯文本。

    实现参考 Qwen 官方示例：last-token pooling、可选 MRL 截断维度、L2 归一化。
    模型与说明见：https://github.com/QwenLM/Qwen3-Embedding
    ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
        embedding_dim: int = 1024,
        instruction: str | None = None,
    ):
        self.model_id = str(model_id).strip()
        self.device_hint = (device or "auto").strip().lower()
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(128, int(max_length))
        self.embedding_dim = int(embedding_dim)
        self.instruction = (
            instruction
            or "Given a web search query, retrieve relevant passages that answer the query"
        )
        self._lock = Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch_device: Any = None

    def _resolve_device(self) -> Any:
        import torch

        if self._torch_device is not None:
            return self._torch_device
        hint = self.device_hint
        if hint in {"", "auto"}:
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif hint == "cpu":
            self._torch_device = torch.device("cpu")
        else:
            self._torch_device = torch.device(hint)
        return self._torch_device

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import is_flash_attn_2_available

            dev = self._resolve_device()
            use_cuda = dev.type == "cuda"
            dtype = torch.float16 if use_cuda else torch.float32
            load_kw: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
            if use_cuda and is_flash_attn_2_available():
                load_kw["attn_implementation"] = "flash_attention_2"
            try:
                model = AutoModel.from_pretrained(self.model_id, **load_kw)
            except Exception:
                load_kw.pop("attn_implementation", None)
                model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=dtype)
            model = model.to(dev)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="left",
            )
            self._model = model
            self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @staticmethod
    def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        import torch
        lp = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
        if lp:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _format_query(self, query: str) -> str:
        return f"Instruct: {self.instruction}\nQuery:{query}"

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        device = self._resolve_device()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if is_query:
                enc_in = [self._format_query(str(t)) for t in batch]
            else:
                enc_in = [str(t) for t in batch]
            inputs = self._tokenizer(
                enc_in,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state
                pooled = self._last_token_pool(hidden, inputs["attention_mask"])
                if self.embedding_dim > 0:
                    pooled = pooled[:, : self.embedding_dim]
                pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.detach().float().cpu().numpy())
        return np.vstack(batches).astype(np.float32)


class ChunkVectorStore:
    """chunk_id 与 embedding 矩阵持久化（.npz），支持追加与清空。"""

    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self._io_lock = Lock()

    def load(self) -> tuple[list[str], np.ndarray]:
        if not self.npz_path.exists():
            return [], np.zeros((0, 1), dtype=np.float32)
        with self._io_lock:
            data = np.load(self.npz_path, allow_pickle=True)
            raw_ids = data["chunk_ids"]
            emb = np.asarray(data["embeddings"], dtype=np.float32)
            chunk_ids = [str(x) for x in raw_ids.tolist()]
            return chunk_ids, emb

    def save(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        with self._io_lock:
            self.npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.npz_path,
                chunk_ids=np.asarray(chunk_ids, dtype=object),
                embeddings=np.asarray(embeddings, dtype=np.float32),
            )

    def append(self, new_ids: list[str], new_vectors: np.ndarray) -> None:
        if not new_ids:
            return
        old_ids, old_emb = self.load()
        if old_emb.size == 0:
            merged_ids = list(new_ids)
            merged_emb = np.asarray(new_vectors, dtype=np.float32)
        else:
            if old_emb.shape[1] != new_vectors.shape[1]:
                raise ValueError(
                    f"向量维度不一致：已有 {old_emb.shape[1]}，新块 {new_vectors.shape[1]}。"
                    "请勿更换 KB_EMBEDDING_MODEL 除非清空知识文档索引。"
                )
            merged_ids = old_ids + list(new_ids)
            merged_emb = np.vstack([old_emb, np.asarray(new_vectors, dtype=np.float32)])
        self.save(merged_ids, merged_emb)

    def clear(self) -> None:
        with self._io_lock:
            if self.npz_path.exists():
                self.npz_path.unlink()


def top_k_cosine(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """query_vec、matrix 行均已 L2 归一化时点积即余弦相似度。"""
    if matrix.size == 0 or k <= 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    sims = matrix @ q
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx_sorted = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx_sorted]


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("entry_type") == "chunk":
                out.append(row)
    return out


def append_chunks_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass
