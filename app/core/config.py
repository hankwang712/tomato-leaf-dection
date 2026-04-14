from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int

    llm_primary_provider: str
    llm_fallback_provider: str
    strict_real_output: bool

    multiagent_llm: str

    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_trust_env: bool

    multiagent_openai_base_url: str
    multiagent_openai_api_key: str
    multiagent_openai_model: str
    multiagent_openai_trust_env: bool

    enable_baseline_report: bool

    local_llm_model_dir: str
    local_llm_max_new_tokens: int
    local_llm_structured_max_new_tokens: int
    local_llm_report_max_new_tokens: int
    local_llm_trust_remote_code: bool
    local_llm_device_map: str
    local_llm_attn_implementation: str
    local_llm_enable_tf32: bool
    local_llm_enable_thinking: bool
    local_llm_structured_decoding_backend: str
    local_llm_structured_decoding_required: bool
    tomato_qa_expert_model_dir: str
    tomato_qa_expert_adapter_model_dir: str

    ollama_base_url: str
    ollama_model: str

    caption_mock_json_path: str
    enable_local_qwen3_vl: bool
    qwen3_vl_model_dir: str
    qwen3_vl_adapter_model_dir: str
    qwen3_vl_max_new_tokens: int
    reranker_base_url: str
    reranker_api_key: str
    reranker_model: str

    run_dir: str
    cases_dir: str
    kb_dir: str
    kb_chunk_max_tokens: int
    kb_chunk_overlap_tokens: int
    kb_embedding_backend: str
    kb_embedding_model: str
    kb_embedding_device: str
    kb_embedding_batch_size: int
    kb_retrieval_candidate_k: int
    kb_use_reranker: bool
    kb_qwen_embedding_dim: int
    kb_qwen_max_length: int
    kb_qwen_query_instruction: str
    enable_local_dinov3: bool
    dinov3_repo_dir: str
    dinov3_backbone_weights: str
    dinov3_classifier_head_weights: str
    dinov3_segmentation_head_weights: str
    dinov3_classes_file: str
    dinov3_classifier_classes_json: str
    dinov3_image_size: int
    dinov3_segmentation_threshold: float

    request_timeout: int
    n_rounds: int
    max_expert_retries: int
    max_parallel_agents_per_layer: int
    max_concurrency_per_route: int
    summary_llm: str
    summary_openai_base_url: str
    summary_openai_api_key: str
    summary_openai_model: str
    summary_max_new_tokens: int

    enable_offline_default: bool
    agent_model_routing_json: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_dir = PROJECT_ROOT / "model"
    default_openai_base = "https://api.openai.com/v1"
    baseline_base = _first_nonempty_env("BASELINE_OPENAI_BASE_URL", "OPENAI_BASE_URL")
    if not baseline_base:
        baseline_base = default_openai_base
    baseline_model = _first_nonempty_env("BASELINE_OPENAI_MODEL", "OPENAI_MODEL")
    if not baseline_model:
        baseline_model = "gpt-4o-mini"
    ma_base = os.getenv("MULTIAGENT_OPENAI_BASE_URL", "").strip()
    ma_key = os.getenv("MULTIAGENT_OPENAI_API_KEY", "").strip()
    return Settings(
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=_get_int("APP_PORT", 8000),
        llm_primary_provider=os.getenv("LLM_PRIMARY_PROVIDER", "openai").strip().lower(),
        llm_fallback_provider=os.getenv("LLM_FALLBACK_PROVIDER", "ollama").strip().lower(),
        strict_real_output=_get_bool("STRICT_REAL_OUTPUT", True),
        multiagent_llm=(
            os.getenv("MULTIAGENT_LLM", "").strip().lower()
            or os.getenv("LLM_PRIMARY_PROVIDER", "openai").strip().lower()
        ),
        openai_base_url=baseline_base,
        openai_api_key=_first_nonempty_env("BASELINE_OPENAI_API_KEY", "OPENAI_API_KEY"),
        openai_model=baseline_model,
        openai_trust_env=(
            _get_bool("BASELINE_OPENAI_TRUST_ENV", False)
            if os.getenv("BASELINE_OPENAI_TRUST_ENV") is not None
            else _get_bool("OPENAI_TRUST_ENV", False)
        ),
        multiagent_openai_base_url=ma_base,
        multiagent_openai_api_key=ma_key,
        multiagent_openai_model=os.getenv("MULTIAGENT_OPENAI_MODEL", "gpt-4o-mini").strip(),
        multiagent_openai_trust_env=_get_bool("MULTIAGENT_OPENAI_TRUST_ENV", False),
        enable_baseline_report=_get_bool("ENABLE_BASELINE_REPORT", True),
        local_llm_model_dir=os.getenv("LOCAL_LLM_MODEL_DIR", "").strip(),
        local_llm_max_new_tokens=_get_int("LOCAL_LLM_MAX_NEW_TOKENS", 2048),
        local_llm_structured_max_new_tokens=_get_int("LOCAL_LLM_STRUCTURED_MAX_NEW_TOKENS", 768),
        local_llm_report_max_new_tokens=_get_int("LOCAL_LLM_REPORT_MAX_NEW_TOKENS", 1400),
        local_llm_trust_remote_code=_get_bool("LOCAL_LLM_TRUST_REMOTE_CODE", True),
        local_llm_device_map=os.getenv("LOCAL_LLM_DEVICE_MAP", "").strip().lower(),
        local_llm_attn_implementation=os.getenv("LOCAL_LLM_ATTN_IMPLEMENTATION", "flash_attention_2").strip(),
        local_llm_enable_tf32=_get_bool("LOCAL_LLM_ENABLE_TF32", True),
        local_llm_enable_thinking=_get_bool("LOCAL_LLM_ENABLE_THINKING", False),
        local_llm_structured_decoding_backend=os.getenv(
            "LOCAL_LLM_STRUCTURED_DECODING_BACKEND",
            "none",
        ).strip(),
        local_llm_structured_decoding_required=_get_bool(
            "LOCAL_LLM_STRUCTURED_DECODING_REQUIRED",
            False,
        ),
        tomato_qa_expert_model_dir=os.getenv("TOMATO_QA_EXPERT_MODEL_DIR", "").strip(),
        tomato_qa_expert_adapter_model_dir=os.getenv(
            "TOMATO_QA_EXPERT_ADAPTER_MODEL_DIR",
            "",
        ).strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip(),
        caption_mock_json_path=os.getenv("CAPTION_MOCK_JSON_PATH", "").strip(),
        enable_local_qwen3_vl=_get_bool("ENABLE_LOCAL_QWEN3_VL", True),
        qwen3_vl_model_dir=os.getenv("QWEN3_VL_MODEL_DIR", str(model_dir / "qwen3-vl-4B")).strip(),
        qwen3_vl_adapter_model_dir=os.getenv("QWEN3_VL_ADAPTER_MODEL_DIR", "").strip(),
        qwen3_vl_max_new_tokens=_get_int("QWEN3_VL_MAX_NEW_TOKENS", 512),
        reranker_base_url=os.getenv("RERANKER_BASE_URL", "").strip(),
        reranker_api_key=os.getenv("RERANKER_API_KEY", "").strip(),
        reranker_model=os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B").strip(),
        run_dir=os.getenv("RUN_DIR", "runs").strip(),
        cases_dir=os.getenv("CASES_DIR", "case_library").strip(),
        kb_dir=os.getenv("KB_DIR", "knowledge_bases").strip(),
        kb_chunk_max_tokens=_get_int("KB_CHUNK_MAX_TOKENS", 500),
        kb_chunk_overlap_tokens=_get_int("KB_CHUNK_OVERLAP_TOKENS", 64),
        kb_embedding_backend=os.getenv("KB_EMBEDDING_BACKEND", "auto").strip().lower(),
        kb_embedding_model=os.getenv(
            "KB_EMBEDDING_MODEL",
            "Qwen/Qwen3-Embedding-4B",
        ).strip(),
        kb_embedding_device=os.getenv("KB_EMBEDDING_DEVICE", "auto").strip(),
        kb_embedding_batch_size=_get_int("KB_EMBEDDING_BATCH_SIZE", 8),
        kb_retrieval_candidate_k=_get_int("KB_RETRIEVAL_CANDIDATE_K", 24),
        kb_use_reranker=_get_bool("KB_USE_RERANKER", False),
        kb_qwen_embedding_dim=(
            (
                -1
                if os.getenv("KB_QWEN_EMBEDDING_DIM", "1024").strip().lower() in {"-1", "full"}
                else _get_int("KB_QWEN_EMBEDDING_DIM", 1024)
            )
        ),
        kb_qwen_max_length=_get_int("KB_QWEN_MAX_LENGTH", 8192),
        kb_qwen_query_instruction=os.getenv(
            "KB_QWEN_QUERY_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query",
        ).strip(),
        enable_local_dinov3=_get_bool("ENABLE_LOCAL_DINOV3", True),
        dinov3_repo_dir=os.getenv("DINOV3_REPO_DIR", str(model_dir / "dinov3")).strip(),
        dinov3_backbone_weights=os.getenv(
            "DINOV3_BACKBONE_WEIGHTS",
            str(model_dir / "dinov3_vitl16_pretrain.pth"),
        ).strip(),
        dinov3_classifier_head_weights=os.getenv(
            "DINOV3_CLASSIFIER_HEAD_WEIGHTS",
            str(model_dir / "best_classifier_head.pth"),
        ).strip(),
        dinov3_segmentation_head_weights=os.getenv(
            "DINOV3_SEGMENTATION_HEAD_WEIGHTS",
            str(model_dir / "best_segmentation_head.pth"),
        ).strip(),
        dinov3_classes_file=os.getenv("DINOV3_CLASSES_FILE", str(model_dir / "classes.txt")).strip(),
        dinov3_classifier_classes_json=os.getenv("DINOV3_CLASSIFIER_CLASSES_JSON", "").strip(),
        dinov3_image_size=_get_int("DINOV3_IMAGE_SIZE", 256),
        dinov3_segmentation_threshold=_get_float("DINOV3_SEGMENTATION_THRESHOLD", 0.5),
        request_timeout=_get_int("REQUEST_TIMEOUT", 60),
        n_rounds=_get_int("N_ROUNDS", 2),
        max_expert_retries=_get_int("MAX_EXPERT_RETRIES", 2),
        max_parallel_agents_per_layer=_get_int("MAX_PARALLEL_AGENTS_PER_LAYER", 3),
        max_concurrency_per_route=_get_int("MAX_CONCURRENCY_PER_ROUTE", 2),
        summary_llm=os.getenv("SUMMARY_LLM", "openai").strip().lower(),
        summary_openai_base_url=os.getenv("SUMMARY_OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        summary_openai_api_key=os.getenv("SUMMARY_OPENAI_API_KEY", "").strip(),
        summary_openai_model=os.getenv("SUMMARY_OPENAI_MODEL", "gpt-4o-mini").strip(),
        summary_max_new_tokens=_get_int("SUMMARY_MAX_NEW_TOKENS", 512),
        enable_offline_default=_get_bool("ENABLE_OFFLINE_DEFAULT", True),
        agent_model_routing_json=os.getenv("AGENT_MODEL_ROUTING_JSON", "").strip(),
    )
