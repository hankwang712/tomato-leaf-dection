from __future__ import annotations

import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from app.core.agents.orchestrator import MultiAgentOrchestrator
from app.core.agents.report_quality import validate_markdown_report
from app.core.agents.sanitizer import sanitize_trace
from app.core.caption.dinov3_caption import build_caption_from_dinov3_analysis
from app.core.caption.presentation import localize_caption_payload
from app.core.caption.provider_http_placeholder import HttpPlaceholderCaptionProvider
from app.core.caption.qwen3_vl_provider import LocalQwen3VLCaptionProvider
from app.core.caption.schema import CaptionSchema
from app.core.config import Settings
from app.core.errors import RealOutputRequiredError
from app.core.llm_clients import build_agent_model_routing, build_llm_client
from app.core.retrieval.faceted_retriever import FacetedRetriever
from app.core.retrieval.knowledge_base import GovernanceKnowledgeBase
from app.core.retrieval.reranker_client import RerankerClient
from app.core.memory import AgentReflection, EpisodicCaseMemory, MemoryConsolidation
from app.core.retrieval.source_router import (
    SourceRouter,
    build_mechanism_hypotheses,
    extract_mechanism_tags,
    extract_mechanism_tags_via_llm,
    summarize_source_alignment,
)
from app.core.storage.case_library import CaseLibrary
from app.core.storage.run_store import RunStore
from app.core.vision.dinov3_service import DinoV3Paths, LocalDinoV3Diagnoser
from app.core.vision.merged_result import build_vision_result
from app.core.vision.presentation import build_image_analysis_display


class DiagnosisPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.run_store = RunStore(settings.run_dir)
        self.case_library = CaseLibrary(settings.cases_dir)
        reranker = RerankerClient(
            base_url=settings.reranker_base_url,
            api_key=settings.reranker_api_key,
            model=settings.reranker_model,
            timeout=settings.request_timeout,
        )
        self.kb = GovernanceKnowledgeBase(
            settings.kb_dir,
            settings=settings,
            reranker=reranker,
        )
        self.retriever = FacetedRetriever(reranker=reranker)
        llm_client = build_llm_client(settings)
        self.source_router = SourceRouter(llm_client=llm_client, timeout=settings.request_timeout)
        self.agent_reflection = AgentReflection(settings.cases_dir)
        self.episodic_memory = EpisodicCaseMemory(settings.cases_dir)
        self.memory_consolidation = MemoryConsolidation(settings.cases_dir)
        self.placeholder_caption_provider = HttpPlaceholderCaptionProvider(
            timeout=settings.request_timeout,
            mock_json_path="" if settings.enable_local_qwen3_vl else settings.caption_mock_json_path,
        )
        self.qwen_caption_provider = self._build_qwen_caption_provider(settings)
        self.image_diagnoser = self._build_image_diagnoser(settings)
        agent_model_routing = build_agent_model_routing(settings)
        self.orchestrator = MultiAgentOrchestrator(
            llm_client=llm_client,
            agent_model_routing=agent_model_routing,
            max_retries=settings.max_expert_retries,
            timeout=settings.request_timeout,
            strict_real_output=settings.strict_real_output,
            max_parallel_agents_per_layer=settings.max_parallel_agents_per_layer,
            max_concurrency_per_route=settings.max_concurrency_per_route,
            structured_max_new_tokens=settings.local_llm_structured_max_new_tokens,
            report_max_new_tokens=settings.local_llm_report_max_new_tokens,
            enable_baseline_report=settings.enable_baseline_report,
            memory_dir=settings.cases_dir,
        )

    def _kb_evidence_query(
        self,
        problem_name: str,
        case_text: str,
        caption: CaptionSchema,
        *,
        for_kb: bool = False,
    ) -> str:
        """构建检索查询.

        Args:
            problem_name: 病名/诊断名（作为核心检索词）
            case_text: 用户可选的当地环境/田间补充说明（表单字段名仍为 case_text）
            caption: 视觉摘要
            for_kb: 是否为知识库检索（更注重病名+症状描述）
        """
        parts = []
        if problem_name:
            parts.append(str(problem_name).strip())
        if for_kb:
            if caption.visual_summary:
                parts.append(str(caption.visual_summary).strip())
            if case_text:
                parts.append(str(case_text).strip()[:200])
        else:
            if case_text:
                parts.append(str(case_text).strip())
            if caption.visual_summary:
                parts.append(str(caption.visual_summary).strip())
            parts.append(self.retriever.build_signature(caption))
        return " ".join(p for p in parts if p).strip()

    def _retrieve_evidence_bundle(
        self,
        *,
        problem_name: str,
        case_text: str,
        caption: CaptionSchema,
        image_display: dict[str, Any] | None,
        vision_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            mechanism_tags = extract_mechanism_tags_via_llm(
                self.source_router.llm_client,
                caption,
                case_text=case_text,
                image_display=image_display,
                vision_result=vision_result,
                timeout=self.settings.request_timeout,
            )
        except Exception:
            mechanism_tags = extract_mechanism_tags(
                caption,
                case_text=case_text,
                image_display=image_display,
                vision_result=vision_result,
            )
        case_query = self._kb_evidence_query(problem_name, case_text, caption)
        kb_query = self._build_kb_specific_query(problem_name, caption)
        case_support = self.case_library.estimate_case_support(case_query, mechanism_tags)
        source_routing = self.source_router.route(
            caption=caption,
            mechanism_tags=mechanism_tags,
            case_support=case_support,
            shared_state={"evidence_gaps": list(caption.followup_questions)},
        )
        budgets = source_routing.get("budgets", {}) if isinstance(source_routing.get("budgets"), dict) else {}
        verified_k = max(1, int(budgets.get("verified_k", 6) or 6))
        unverified_k = max(1, int(budgets.get("unverified_k", 4) or 4))
        document_k = max(1, int(budgets.get("document_k", 6) or 6))

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_verified = executor.submit(
                self.case_library.retrieve_text,
                case_query,
                True,
                verified_k,
                mechanism_tags,
            )
            future_unverified = executor.submit(
                self.case_library.retrieve_text,
                case_query,
                False,
                unverified_k,
                mechanism_tags,
            )
            future_kb = executor.submit(self.kb.retrieve_documents, kb_query, document_k)
            verified_records = future_verified.result()
            unverified_records = future_unverified.result()
            document_records = future_kb.result()

        for doc in document_records:
            doc["entry_type"] = "chunk"

        source_agreement, source_conflicts = summarize_source_alignment(
            caption=caption,
            mechanism_tags=mechanism_tags,
            source_routing=source_routing,
            case_records=verified_records + unverified_records,
            document_records=document_records,
        )
        source_routing = {
            **source_routing,
            "source_agreement": source_agreement,
            "source_conflicts": source_conflicts,
            "mechanism_hypotheses": build_mechanism_hypotheses(mechanism_tags),
            "retrieved_counts": {
                "verified_cases": len(verified_records),
                "unverified_cases": len(unverified_records),
                "documents": len(document_records),
            },
        }
        kb_evidence = self.retriever.retrieve(
            caption=caption,
            candidates=verified_records + unverified_records + document_records,
            k=4,
            query_text=case_query,
            mechanism_tags=mechanism_tags,
            source_routing=source_routing,
        )
        return {
            "case_query": case_query,
            "kb_query": kb_query,
            "mechanism_tags": mechanism_tags,
            "source_routing": source_routing,
            "kb_evidence": kb_evidence,
        }

    def _build_memory_profile(
        self,
        *,
        caption: CaptionSchema,
        trace: dict[str, Any],
        safety_payload: dict[str, Any],
        case_quality: dict[str, Any],
        mechanism_tags: dict[str, Any],
        source_routing: dict[str, Any],
    ) -> dict[str, Any]:
        shared_state = trace.get("shared_state", {}) if isinstance(trace.get("shared_state"), dict) else {}
        uncertainty_score = float(shared_state.get("uncertainty_score", 1.0) or 1.0)
        quorum_score = float(shared_state.get("quorum_score", 0.0) or 0.0)
        inhibition_score = float(shared_state.get("inhibition_score", 1.0) or 1.0)
        source_conflicts = self.orchestrator._unique_strings(shared_state.get("source_conflicts", []))
        memory_score = self.orchestrator._clamp(
            (0.24 if bool(safety_payload.get("safety_passed", False)) else 0.0)
            + (1.0 - uncertainty_score) * 0.22
            + quorum_score * 0.18
            + (1.0 - inhibition_score) * 0.16
            + (1.0 - float(caption.ood_score)) * 0.10
            + (0.06 if case_quality.get("has_citation") else 0.0)
            + (0.06 if case_quality.get("has_supporting_evidence") else 0.0)
        )
        memory_reasons: list[str] = []
        if bool(safety_payload.get("safety_passed", False)):
            memory_reasons.append("safety_passed")
        if uncertainty_score <= 0.35:
            memory_reasons.append("low_uncertainty")
        if inhibition_score <= MultiAgentOrchestrator.INHIBITION_BLOCK_THRESHOLD:
            memory_reasons.append("low_inhibition")
        if float(caption.ood_score) <= 0.25:
            memory_reasons.append("low_ood")
        if case_quality.get("has_citation"):
            memory_reasons.append("has_valid_citation")
        if case_quality.get("has_supporting_evidence"):
            memory_reasons.append("has_supporting_evidence")
        if not source_conflicts:
            memory_reasons.append("low_source_conflict")
        if mechanism_tags.get("stressor_class") and mechanism_tags.get("stressor_class") != ["unknown"]:
            memory_reasons.append("mechanism_tags_available")
        if source_routing.get("mode") == "case_priority":
            memory_reasons.append("case_priority_route")

        is_reference = (
            bool(safety_payload.get("safety_passed", False))
            and case_quality.get("write_verified", False)
            and uncertainty_score <= 0.35
            and inhibition_score <= MultiAgentOrchestrator.INHIBITION_BLOCK_THRESHOLD
            and not source_conflicts
            and float(caption.ood_score) <= 0.25
            and memory_score >= 0.66
        )
        return {
            "memory_level": "reference_memory" if is_reference else "candidate_memory",
            "memory_score": round(float(memory_score), 4),
            "memory_reasons": memory_reasons,
            "quorum_score": round(float(quorum_score), 4),
            "inhibition_score": round(float(inhibition_score), 4),
        }

    def _build_kb_specific_query(self, problem_name: str, caption: CaptionSchema) -> str:
        """专为知识库检索构建查询，强调病名和症状特征."""
        parts = [str(problem_name or "").strip()]
        s = caption.symptoms
        symptom_keywords = []
        if s.color:
            symptom_keywords.extend([x.value for x in s.color if x.value])
        if s.spot_shape:
            symptom_keywords.extend([x.value for x in s.spot_shape if x.value])
        if s.tissue_state:
            symptom_keywords.extend([x.value for x in s.tissue_state if x.value])
        if s.distribution_position:
            symptom_keywords.extend([x.value for x in s.distribution_position if x.value])
        if symptom_keywords:
            parts.append(" ".join(symptom_keywords))
        if caption.visual_summary:
            parts.append(str(caption.visual_summary).strip())
        return " ".join(p for p in parts if p).strip()

    def run(
        self,
        problem_name: str,
        case_text: str,
        stage: str = "initial",
        image_bytes: bytes | None = None,
        n_rounds: int | None = None,
        image_filename: str = "",
        image_content_type: str = "",
    ) -> dict[str, Any]:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        rounds_to_use = n_rounds or self.settings.n_rounds
        image_meta = self._build_image_asset_meta(
            run_id,
            image_bytes,
            image_filename=image_filename,
            image_content_type=image_content_type,
        )

        input_payload = {
            "run_id": run_id,
            "problem_name": problem_name,
            "case_text": case_text,
            "stage": stage,
            "n_rounds": rounds_to_use,
            "timestamp": datetime.now().isoformat(),
            "source_image": image_meta,
        }

        try:
            image_analysis = self._analyze_image(image_bytes=image_bytes)
            slot_extraction = self._extract_slot_extraction(case_text=case_text, image_bytes=image_bytes)
            caption = self._build_caption(
                case_text=case_text,
                image_bytes=image_bytes,
                image_analysis=image_analysis,
                slot_extraction=slot_extraction,
            )
            caption_data = localize_caption_payload(caption.model_dump(mode="json"))
            image_display = build_image_analysis_display(image_analysis) if image_analysis else {}
            vision_result = (
                build_vision_result(
                    slot_extraction=slot_extraction,
                    image_analysis=image_analysis,
                    display=image_display,
                    caption=caption_data,
                )
                if image_analysis is not None or slot_extraction is not None
                else None
            )

            evidence_bundle = self._retrieve_evidence_bundle(
                problem_name=problem_name,
                case_text=case_text,
                caption=caption,
                image_display=image_display,
                vision_result=vision_result,
            )
            mechanism_tags = evidence_bundle["mechanism_tags"]
            source_routing = evidence_bundle["source_routing"]
            kb_evidence = evidence_bundle["kb_evidence"]

            raw_result = self.orchestrator.run(
                case_text=case_text,
                caption=caption,
                kb_evidence=kb_evidence,
                n_rounds=rounds_to_use,
                vision_result=vision_result,
                mechanism_tags=mechanism_tags,
                source_routing=source_routing,
            )
            return self._finalize_and_save(
                run_id=run_id,
                problem_name=problem_name,
                case_text=case_text,
                stage=stage,
                n_rounds=rounds_to_use,
                input_payload=input_payload,
                image_bytes=image_bytes,
                image_meta=image_meta,
                caption=caption,
                caption_data=caption_data,
                image_analysis=image_analysis,
                image_display=image_display,
                vision_result=vision_result,
                slot_extraction=slot_extraction,
                kb_evidence=kb_evidence,
                raw_result=raw_result,
                mechanism_tags=mechanism_tags,
                source_routing=source_routing,
            )
        except RealOutputRequiredError as err:
            self._save_error_log(
                error=err,
                problem_name=problem_name,
                case_text=case_text,
                stage=stage,
            )
            raise

    def run_stream(
        self,
        problem_name: str,
        case_text: str,
        stage: str = "initial",
        image_bytes: bytes | None = None,
        n_rounds: int | None = None,
        image_filename: str = "",
        image_content_type: str = "",
    ) -> Iterator[dict[str, Any]]:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        rounds_to_use = n_rounds or self.settings.n_rounds
        image_meta = self._build_image_asset_meta(
            run_id,
            image_bytes,
            image_filename=image_filename,
            image_content_type=image_content_type,
        )

        input_payload = {
            "run_id": run_id,
            "problem_name": problem_name,
            "case_text": case_text,
            "stage": stage,
            "n_rounds": rounds_to_use,
            "timestamp": datetime.now().isoformat(),
            "source_image": image_meta,
        }
        yield {
            "type": "run_started",
            "run_id": run_id,
            "problem_name": problem_name,
            "case_text": case_text,
            "stage": stage,
            "n_rounds": rounds_to_use,
        }
        if image_bytes:
            yield {
                "type": "image_processing_started",
                "run_id": run_id,
                "message": "已接收图片，正在执行 Qwen3-VL 槽位抽取和 DINOv3 分类分割。",
            }

        try:
            image_analysis = self._analyze_image(image_bytes=image_bytes)
            slot_extraction = self._extract_slot_extraction(case_text=case_text, image_bytes=image_bytes)
            caption = self._build_caption(
                case_text=case_text,
                image_bytes=image_bytes,
                image_analysis=image_analysis,
                slot_extraction=slot_extraction,
            )
            caption_data = localize_caption_payload(caption.model_dump(mode="json"))
            image_display = build_image_analysis_display(image_analysis) if image_analysis else {}
            vision_result = (
                build_vision_result(
                    slot_extraction=slot_extraction,
                    image_analysis=image_analysis,
                    display=image_display,
                    caption=caption_data,
                )
                if image_analysis is not None or slot_extraction is not None
                else None
            )
            yield {
                "type": "caption_ready",
                "run_id": run_id,
                "caption": caption_data,
                "slot_extraction": slot_extraction or {},
                "image_analysis": image_analysis or {},
                "image_analysis_display": image_display,
                "vision_result": vision_result,
            }
            if slot_extraction is not None:
                yield {"type": "slot_extraction_ready", "run_id": run_id, "slot_extraction": slot_extraction}
            if image_analysis is not None:
                yield {
                    "type": "image_analysis_ready",
                    "run_id": run_id,
                    "image_analysis": image_analysis,
                    "display": image_display,
                }

            evidence_bundle = self._retrieve_evidence_bundle(
                problem_name=problem_name,
                case_text=case_text,
                caption=caption,
                image_display=image_display,
                vision_result=vision_result,
            )
            mechanism_tags = evidence_bundle["mechanism_tags"]
            source_routing = evidence_bundle["source_routing"]
            kb_evidence = evidence_bundle["kb_evidence"]
            yield {
                "type": "kb_ready",
                "run_id": run_id,
                "kb_evidence_count": len(kb_evidence),
                "kb_evidence": kb_evidence,
                "mechanism_tags": mechanism_tags,
                "source_routing": source_routing,
            }

            raw_result: dict[str, Any] | None = None
            for event in self.orchestrator.run_iter(
                case_text=case_text,
                caption=caption,
                kb_evidence=kb_evidence,
                n_rounds=rounds_to_use,
                vision_result=vision_result,
                mechanism_tags=mechanism_tags,
                source_routing=source_routing,
            ):
                if event.get("type") == "orchestrator_complete":
                    result = event.get("result")
                    if isinstance(result, dict):
                        raw_result = result
                    continue
                event["run_id"] = run_id
                yield event

            if not isinstance(raw_result, dict):
                raise RealOutputRequiredError(
                    stage="orchestrator_complete",
                    agent_name="system",
                    provider="",
                    model="",
                    reason="编排器未产出最终结果",
                    raw_error_type="RuntimeError",
                )

            yield {"type": "reports_started", "run_id": run_id}
            result = self._finalize_and_save(
                run_id=run_id,
                problem_name=problem_name,
                case_text=case_text,
                stage=stage,
                n_rounds=rounds_to_use,
                input_payload=input_payload,
                image_bytes=image_bytes,
                image_meta=image_meta,
                caption=caption,
                caption_data=caption_data,
                image_analysis=image_analysis,
                image_display=image_display,
                vision_result=vision_result,
                slot_extraction=slot_extraction,
                kb_evidence=kb_evidence,
                raw_result=raw_result,
                mechanism_tags=mechanism_tags,
                source_routing=source_routing,
            )
            yield {
                "type": "reports_ready",
                "run_id": run_id,
                "multi_agent_meta": result["reports"].get("multi_agent_meta", {}),
                "baseline_meta": result["reports"].get("baseline_meta", {}),
            }
            yield {"type": "complete", "run_id": run_id, "result": result}
        except RealOutputRequiredError as err:
            self._save_error_log(
                error=err,
                problem_name=problem_name,
                case_text=case_text,
                stage=stage,
            )
            yield {"type": "error", "run_id": run_id, "detail": err.to_detail()}
        except Exception as err:  # noqa: BLE001
            detail = {
                "code": "PIPELINE_STREAM_ERROR",
                "stage": "run_stream",
                "agent_name": "system",
                "provider": "",
                "model": "",
                "message": str(err),
                "error_type": type(err).__name__,
            }
            yield {"type": "error", "run_id": run_id, "detail": detail}

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.run_store.list_runs(limit=limit)

    def list_cases(self) -> dict[str, Any]:
        return self.case_library.load_all_cases()

    def delete_case(self, run_id: str) -> dict[str, Any]:
        removed = self.case_library.delete_by_run_id(run_id)
        if int(removed.get("removed_total", 0)) <= 0:
            raise FileNotFoundError(run_id)
        return {
            "ok": True,
            "run_id": run_id,
            **removed,
        }

    def delete_run(self, run_id: str) -> dict[str, Any]:
        removed_run = self.run_store.delete_run(run_id)
        removed_cases = self.case_library.delete_by_run_id(run_id)
        if not removed_run:
            raise FileNotFoundError(run_id)
        return {
            "ok": True,
            "run_id": run_id,
            "removed_case_records": removed_cases.get("removed_total", 0),
        }

    def load_final(self, run_id: str) -> dict[str, Any]:
        return self.run_store.load_json(run_id, "final.json")

    def load_trace(self, run_id: str) -> dict[str, Any]:
        return self.run_store.load_json(run_id, "trace.json")

    def get_run_image_path(self, run_id: str, filename: str) -> Path:
        return self.run_store.load_file_path(run_id, filename)

    def clear_knowledge_base(self, target: str = "all") -> dict[str, Any]:
        return self.kb.clear_cases(target=target)

    def inspect_image(self, image_bytes: bytes, case_text: str = "") -> dict[str, Any]:
        image_analysis = self._analyze_image(image_bytes=image_bytes)
        if image_analysis is None:
            raise RuntimeError("本地 DINOv3 图像分析不可用")
        slot_extraction = self._extract_slot_extraction(case_text=case_text, image_bytes=image_bytes)
        caption = self._build_caption(
            case_text=case_text,
            image_bytes=image_bytes,
            image_analysis=image_analysis,
            slot_extraction=slot_extraction,
        )
        caption_data = localize_caption_payload(caption.model_dump(mode="json"))
        display = build_image_analysis_display(image_analysis)
        vision_result = build_vision_result(
            slot_extraction=slot_extraction,
            image_analysis=image_analysis,
            display=display,
            caption=caption_data,
        )
        return {
            "slot_extraction": slot_extraction or {},
            "image_analysis": image_analysis,
            "display": display,
            "caption": caption_data,
            "vision_result": vision_result,
        }

    def _build_image_asset_meta(
        self,
        run_id: str,
        image_bytes: bytes | None,
        *,
        image_filename: str = "",
        image_content_type: str = "",
    ) -> dict[str, Any]:
        if not image_bytes:
            return {}
        original_name = Path(image_filename or "").name.strip()
        suffix = Path(original_name).suffix.lower()
        if not suffix and image_content_type:
            suffix = mimetypes.guess_extension(image_content_type.split(";")[0].strip()) or ""
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            suffix = ".png"
        base_name = Path(original_name).stem.strip() or "uploaded-image"
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._-") or "uploaded-image"
        stored_filename = f"source_image{suffix}"
        return {
            "original_filename": f"{safe_base}{suffix}",
            "stored_filename": stored_filename,
            "content_type": image_content_type.split(";")[0].strip() or self.run_store.guess_media_type(stored_filename),
            "size_bytes": len(image_bytes),
            "url": f"/api/v1/runs/{run_id}/image/{stored_filename}",
        }

    def _build_qwen_caption_provider(self, settings: Settings) -> LocalQwen3VLCaptionProvider | None:
        if not settings.enable_local_qwen3_vl:
            return None
        adapter_dir = settings.qwen3_vl_adapter_model_dir.strip() or None
        provider = LocalQwen3VLCaptionProvider(
            model_dir=settings.qwen3_vl_model_dir,
            adapter_model_dir=adapter_dir,
            max_new_tokens=settings.qwen3_vl_max_new_tokens,
            prefer_cuda=True,
            timeout=settings.request_timeout,
        )
        return provider if provider.is_available() else None

    def _build_image_diagnoser(self, settings: Settings) -> LocalDinoV3Diagnoser | None:
        if not settings.enable_local_dinov3:
            return None
        classifier_classes: tuple[str, ...] = ()
        raw_classifier_classes = settings.dinov3_classifier_classes_json.strip()
        if raw_classifier_classes:
            parsed = json.loads(raw_classifier_classes)
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise RuntimeError("DINOV3_CLASSIFIER_CLASSES_JSON 必须是字符串数组 JSON")
            classifier_classes = tuple(item.strip() for item in parsed if item.strip())
        return LocalDinoV3Diagnoser(
            DinoV3Paths(
                repo_dir=Path(settings.dinov3_repo_dir),
                backbone_weights=Path(settings.dinov3_backbone_weights),
                classifier_head_weights=Path(settings.dinov3_classifier_head_weights),
                segmentation_head_weights=Path(settings.dinov3_segmentation_head_weights),
                classes_file=Path(settings.dinov3_classes_file),
                classifier_classes=classifier_classes,
                image_size=settings.dinov3_image_size,
                segmentation_threshold=settings.dinov3_segmentation_threshold,
                prefer_cuda=True,
            )
        )

    def _migrate_legacy_cases_if_needed(self) -> None:
        return None

    def _analyze_image(self, image_bytes: bytes | None) -> dict[str, Any] | None:
        if not image_bytes or self.image_diagnoser is None:
            return None
        if not self.image_diagnoser.is_available():
            return None
        return self.image_diagnoser.analyze_image_bytes(image_bytes)

    def _extract_slot_extraction(
        self,
        *,
        case_text: str,
        image_bytes: bytes | None,
    ) -> dict[str, Any] | None:
        if not image_bytes or self.qwen_caption_provider is None:
            return None
        try:
            return self.qwen_caption_provider.extract_slots(case_text=case_text, image_bytes=image_bytes)
        except Exception:
            return None

    def _build_caption(
        self,
        case_text: str,
        image_bytes: bytes | None,
        image_analysis: dict[str, Any] | None = None,
        slot_extraction: dict[str, Any] | None = None,
    ) -> CaptionSchema:
        if slot_extraction is not None and self.qwen_caption_provider is not None:
            fallback_caption = self.qwen_caption_provider.caption_from_slots(slot_extraction, case_text=case_text)
        elif image_analysis is None:
            return self.placeholder_caption_provider.predict(case_text=case_text, image_bytes=image_bytes)
        else:
            fallback_caption = self.placeholder_caption_provider.predict(case_text=case_text, image_bytes=None)
        if image_analysis is None:
            return fallback_caption
        return build_caption_from_dinov3_analysis(
            case_text=case_text,
            image_analysis=image_analysis,
            fallback_caption=fallback_caption,
        )

    def _finalize_and_save(
        self,
        run_id: str,
        problem_name: str,
        case_text: str,
        stage: str,
        n_rounds: int,
        input_payload: dict[str, Any],
        image_bytes: bytes | None,
        image_meta: dict[str, Any],
        caption: CaptionSchema,
        caption_data: dict[str, Any],
        image_analysis: dict[str, Any] | None,
        image_display: dict[str, Any] | None,
        vision_result: dict[str, Any] | None,
        slot_extraction: dict[str, Any] | None,
        kb_evidence: list[dict[str, Any]],
        raw_result: dict[str, Any],
        mechanism_tags: dict[str, Any],
        source_routing: dict[str, Any],
    ) -> dict[str, Any]:
        clean_trace = sanitize_trace(raw_result)
        shared_state = clean_trace.get("shared_state", {}) if isinstance(clean_trace.get("shared_state"), dict) else {}
        shared_state["mechanism_tags"] = mechanism_tags
        shared_state["source_routing"] = source_routing
        shared_state["source_agreement"] = source_routing.get("source_agreement", [])
        shared_state["source_conflicts"] = source_routing.get("source_conflicts", [])
        shared_state["mechanism_hypotheses"] = source_routing.get("mechanism_hypotheses", [])
        if "action_gate" not in shared_state:
            shared_state["action_gate"] = "balanced"
        clean_trace["shared_state"] = shared_state
        clean_trace["caption"] = caption_data
        if slot_extraction is not None:
            clean_trace["slot_extraction"] = slot_extraction
        if image_analysis is not None:
            clean_trace["image_analysis"] = image_analysis
            clean_trace["image_analysis_display"] = image_display or build_image_analysis_display(image_analysis)
        if vision_result is not None:
            clean_trace["vision_result"] = vision_result
        if isinstance(raw_result.get("decision_packet"), dict):
            clean_trace["decision_packet"] = raw_result["decision_packet"]
        clean_trace["kb_evidence"] = kb_evidence
        clean_trace["mechanism_tags"] = mechanism_tags
        clean_trace["source_routing"] = source_routing
        safety_payload = raw_result.get(
            "safety",
            {
                "safety_passed": True,
                "flags": [],
                "revised_actions": [],
                "action_level": "fully_supported",
                "review_summary": "",
                "prohibited_actions": [],
                "required_followups": [],
                "evidence_sufficiency": "",
            },
        )
        case_quality = self.case_library.evaluate_case_quality(clean_trace, safety_payload)
        memory_profile = self._build_memory_profile(
            caption=caption,
            trace=clean_trace,
            safety_payload=safety_payload,
            case_quality=case_quality,
            mechanism_tags=mechanism_tags,
            source_routing=source_routing,
        )
        clean_trace.update(memory_profile)
        clean_trace["source_agreement"] = source_routing.get("source_agreement", [])
        clean_trace["source_conflicts"] = source_routing.get("source_conflicts", [])
        if self.settings.strict_real_output:
            self._assert_trace_real_output(clean_trace)

        top_nm = str(clean_trace.get("final", {}).get("top_diagnosis", {}).get("name", "") or "").strip()
        if top_nm:
            kb_query = f"{top_nm} {case_text}"[:600].strip()
        else:
            kb_query = case_text[:600].strip() if case_text else ""
        kb_docs = self.kb.retrieve_documents(kb_query, k=4) if kb_query else []
        for doc in kb_docs:
            doc["entry_type"] = "chunk"

        reports_bundle = self.orchestrator.generate_reports(
            case_text=case_text,
            caption=caption,
            kb_evidence=kb_evidence,
            rounds=clean_trace["rounds"],
            final_result=clean_trace["final"],
            safety_result=safety_payload,
            vision_result=vision_result,
            image_bytes=image_bytes,
            kb_documents=kb_docs,
        )
        report_quality_issues = self._collect_report_quality_issues(
            multi_agent_markdown=reports_bundle["multi_agent_markdown"],
            baseline_markdown=reports_bundle["baseline_markdown"],
            baseline_error=reports_bundle.get("baseline_error"),
            baseline_skipped=bool(reports_bundle.get("baseline_disabled")),
        )

        clean_trace["report_meta"] = {
            "multi_agent_meta": reports_bundle["multi_agent_meta"],
            "baseline_meta": reports_bundle["baseline_meta"],
            "quality_issues": report_quality_issues,
        }
        if isinstance(reports_bundle.get("baseline_error"), dict):
            clean_trace["report_meta"]["baseline_error"] = reports_bundle["baseline_error"]
        if reports_bundle.get("baseline_disabled"):
            clean_trace["report_meta"]["baseline_disabled"] = True

        comparison_summary = self._build_comparison_summary(
            final_result=clean_trace["final"],
            baseline_result=reports_bundle["baseline_structured"],
            baseline_disabled=bool(reports_bundle.get("baseline_disabled")),
        )
        reports_payload = {
            "multi_agent_markdown": reports_bundle["multi_agent_markdown"],
            "baseline_markdown": reports_bundle["baseline_markdown"],
            "multi_agent_meta": reports_bundle["multi_agent_meta"],
            "baseline_meta": reports_bundle["baseline_meta"],
            "quality_issues": report_quality_issues,
        }
        if reports_bundle.get("baseline_disabled"):
            reports_payload["baseline_disabled"] = True
        if isinstance(reports_bundle.get("baseline_error"), dict):
            reports_payload["baseline_error"] = reports_bundle["baseline_error"]
        if isinstance(reports_bundle.get("report_packet"), dict):
            reports_payload["report_packet"] = reports_bundle["report_packet"]
        final_payload = {
            "run_id": run_id,
            "problem_name": problem_name,
            "case_text": case_text,
            "stage": stage,
            "n_rounds": n_rounds,
            "source_image": image_meta,
            **clean_trace["final"],
            "reports_ref": "reports.json",
            "reports": reports_payload,
            "comparison_summary": comparison_summary,
            "shared_state": clean_trace.get("shared_state", {}),
            "execution_meta": clean_trace.get("execution_meta", {}),
            "mechanism_tags": mechanism_tags,
            "source_routing": source_routing,
            "memory_level": memory_profile["memory_level"],
            "memory_score": memory_profile["memory_score"],
            "memory_reasons": memory_profile["memory_reasons"],
            "quorum_score": memory_profile["quorum_score"],
            "inhibition_score": memory_profile["inhibition_score"],
            "source_agreement": source_routing.get("source_agreement", []),
            "source_conflicts": source_routing.get("source_conflicts", []),
        }
        if isinstance(clean_trace.get("decision_packet"), dict):
            final_payload["decision_packet"] = clean_trace["decision_packet"]
        if isinstance(reports_bundle.get("report_packet"), dict):
            final_payload["report_packet"] = reports_bundle["report_packet"]
        if image_analysis is not None:
            final_payload["image_analysis"] = image_analysis
            final_payload["image_analysis_display"] = image_display or build_image_analysis_display(image_analysis)
        if slot_extraction is not None:
            final_payload["slot_extraction"] = slot_extraction
        if vision_result is not None:
            final_payload["vision_result"] = vision_result

        if image_bytes and image_meta.get("stored_filename"):
            self.run_store.save_bytes(run_id, image_meta["stored_filename"], image_bytes)
        self.run_store.save_json(run_id, "input.json", input_payload)
        self.run_store.save_json(run_id, "caption.json", caption_data)
        self.run_store.save_json(run_id, "trace.json", clean_trace)
        self.run_store.save_json(run_id, "final.json", final_payload)
        self.run_store.save_json(run_id, "safety.json", safety_payload)
        self.run_store.save_json(run_id, "reports.json", reports_payload)

        case_record = self._build_case_record(
            run_id=run_id,
            problem_name=problem_name,
            case_text=case_text,
            caption_data=caption_data,
            final_payload=final_payload,
            reports_payload=reports_payload,
            safety_payload=safety_payload,
            image_display=image_display,
            vision_result=vision_result,
            mechanism_tags=mechanism_tags,
            source_routing=source_routing,
            memory_profile=memory_profile,
        )
        write_verified = bool(case_quality.get("write_verified", False))
        self.case_library.save_case(case_record, verified=write_verified)

        self._store_episodic_memory(
            run_id=run_id,
            caption=caption,
            mechanism_tags=mechanism_tags,
            clean_trace=clean_trace,
            final_payload=final_payload,
            safety_payload=safety_payload,
        )

        return {
            "run_id": run_id,
            "final": final_payload,
            "trace": clean_trace,
            "reports": reports_payload,
            "execution_meta": clean_trace.get("execution_meta", {}),
            "safety": safety_payload,
            "case_write_layer": "verified" if write_verified else "unverified",
            "knowledge_write_layer": "verified" if write_verified else "unverified",
        }

    def _build_case_record(
        self,
        *,
        run_id: str,
        problem_name: str,
        case_text: str,
        caption_data: dict[str, Any],
        final_payload: dict[str, Any],
        reports_payload: dict[str, Any],
        safety_payload: dict[str, Any],
        image_display: dict[str, Any] | None,
        vision_result: dict[str, Any] | None,
        mechanism_tags: dict[str, Any],
        source_routing: dict[str, Any],
        memory_profile: dict[str, Any],
    ) -> dict[str, Any]:
        top_diagnosis = final_payload.get("top_diagnosis", {}) if isinstance(final_payload, dict) else {}
        top_name = str(top_diagnosis.get("name", "")).strip() or "未给出主诊断"
        confidence = str(top_diagnosis.get("confidence", "")).strip()
        visual_summary = str(caption_data.get("visual_summary", "")).strip()
        image_summary = ""
        if isinstance(image_display, dict):
            image_summary = str(image_display.get("摘要", "")).strip()
        conflict_summary = ""
        if isinstance(vision_result, dict):
            conflict = vision_result.get("conflict_analysis", {})
            if isinstance(conflict, dict):
                conflict_summary = str(conflict.get("reason_summary", "")).strip()

        summary_parts = [
            f"主诊断倾向：{top_name}" + (f"（置信说明：{confidence}）" if confidence else ""),
            image_summary,
            visual_summary,
            conflict_summary,
            str(final_payload.get("confidence_statement", "")).strip(),
        ]
        case_summary = " ".join(part for part in summary_parts if part)

        return {
            "run_id": run_id,
            "problem_name": problem_name,
            "case_text": case_text,
            "case_summary": case_summary,
            "caption": caption_data,
            "image_analysis_display": image_display or {},
            "top_diagnosis": top_diagnosis,
            "candidates": final_payload.get("candidates", []),
            "actions": final_payload.get("actions", []),
            "rescue_plan": final_payload.get("rescue_plan", []),
            "evidence_sufficiency": final_payload.get("evidence_sufficiency", ""),
            "confidence_statement": final_payload.get("confidence_statement", ""),
            "report_outline": final_payload.get("report_outline", []),
            "report_summary": case_summary,
            "report_excerpt": str(reports_payload.get("multi_agent_markdown", "")).strip()[:1200],
            "safety_passed": bool(safety_payload.get("safety_passed", False)),
            "memory_level": memory_profile.get("memory_level", "candidate_memory"),
            "memory_score": memory_profile.get("memory_score", 0.0),
            "memory_reasons": memory_profile.get("memory_reasons", []),
            "mechanism_tags": mechanism_tags,
            "source_routing": source_routing,
            "quorum_score": memory_profile.get("quorum_score", 0.0),
            "inhibition_score": memory_profile.get("inhibition_score", 0.0),
            "source_agreement": source_routing.get("source_agreement", []),
            "source_conflicts": source_routing.get("source_conflicts", []),
            "timestamp": datetime.now().isoformat(),
        }

    def _store_episodic_memory(
        self,
        *,
        run_id: str,
        caption: CaptionSchema,
        mechanism_tags: dict[str, Any],
        clean_trace: dict[str, Any],
        final_payload: dict[str, Any],
        safety_payload: dict[str, Any],
    ) -> None:
        """将诊断过程存储为情景记忆，并记录 Agent 反思。"""
        try:
            top_diag = str(final_payload.get("top_diagnosis", {}).get("name", "")).strip()
            confidence = str(final_payload.get("confidence_statement", "")).strip()
            shared_state = clean_trace.get("shared_state", {})

            visual_sig = {
                "color": [v.value for v in caption.symptoms.color],
                "tissue_state": [v.value for v in caption.symptoms.tissue_state],
                "spot_shape": [v.value for v in caption.symptoms.spot_shape],
                "boundary": [v.value for v in caption.symptoms.boundary],
                "distribution_position": [v.value for v in caption.symptoms.distribution_position],
                "distribution_pattern": [v.value for v in caption.symptoms.distribution_pattern],
                "morph_change": [v.value for v in caption.symptoms.morph_change],
                "co_signs": [v.value for v in caption.symptoms.co_signs],
            }

            diagnosis_path = []
            for rnd in clean_trace.get("rounds", []):
                diagnosis_path.append({
                    "round": rnd.get("round", 0),
                    "consensus": list(rnd.get("summary", {}).get("consensus", []))[:3],
                    "conflicts": list(rnd.get("summary", {}).get("conflicts", []))[:3],
                })

            self.episodic_memory.store_episode(
                run_id=run_id,
                visual_signature=visual_sig,
                mechanism_tags=mechanism_tags,
                diagnosis_path=diagnosis_path,
                final_diagnosis=top_diag,
                confidence_level=confidence,
                agent_consensus={
                    "agreed": list(shared_state.get("consensus", []))[:5],
                    "disagreed": list(shared_state.get("conflicts", []))[:5],
                },
                key_evidence=list(shared_state.get("source_agreement", []))[:5],
                key_conflicts=list(shared_state.get("source_conflicts", []))[:5],
                resolution_strategy=str(shared_state.get("action_gate", "balanced")),
                outcome_quality="high" if safety_payload.get("safety_passed") else "low",
            )

            for rnd in clean_trace.get("rounds", []):
                for turn in rnd.get("expert_turns", []):
                    agent_name = str(turn.get("agent_name", "")).strip()
                    if not agent_name:
                        continue
                    self.agent_reflection.record_reflection(
                        agent_name=agent_name,
                        run_id=run_id,
                        diagnosis_outcome=top_diag,
                        agent_contribution=str(turn.get("evidence_strength", "")).strip()
                        or str(turn.get("role", "")).strip()[:100],
                        behavioral_notes=[],
                        accuracy_signal="unknown",
                    )
        except Exception:
            pass

    def _assert_trace_real_output(self, trace: dict[str, Any]) -> None:
        for round_item in trace.get("rounds", []):
            for turn in round_item.get("expert_turns", []):
                if bool(turn.get("invalid_turn", False)):
                    raise RealOutputRequiredError(
                        stage="trace_validation",
                        agent_name=str(turn.get("agent_name", "")),
                        provider="",
                        model="",
                        reason="检测到 invalid_turn 标记",
                        raw_error_type="ValueError",
                    )
                meta = turn.get("meta", {})
                if not isinstance(meta, dict) or not bool(meta.get("is_real_output", False)):
                    raise RealOutputRequiredError(
                        stage="trace_validation",
                        agent_name=str(turn.get("agent_name", "")),
                        provider=str(meta.get("provider", "")) if isinstance(meta, dict) else "",
                        model=str(meta.get("model", "")) if isinstance(meta, dict) else "",
                        reason="turn 元信息缺失或无效（meta）",
                        raw_error_type="ValueError",
                    )
                for citation in turn.get("citations", []):
                    if str(citation).startswith("fallback_"):
                        raise RealOutputRequiredError(
                            stage="trace_validation",
                            agent_name=str(turn.get("agent_name", "")),
                            provider=str(meta.get("provider", "")),
                            model=str(meta.get("model", "")),
                            reason="检测到 fallback 引用标记",
                            raw_error_type="ValueError",
                        )

        top_name = str(trace.get("final", {}).get("top_diagnosis", {}).get("name", "")).strip().lower()
        if not top_name or top_name == "unknown":
            raise RealOutputRequiredError(
                stage="trace_validation",
                agent_name="coordinator_final",
                provider="",
                model="",
                reason="final.top_diagnosis.name 为空或未知",
                raw_error_type="ValueError",
            )

    def _build_comparison_summary(
        self,
        final_result: dict[str, Any],
        baseline_result: dict[str, Any],
        *,
        baseline_disabled: bool = False,
    ) -> dict[str, Any]:
        multi_top = final_result.get("top_diagnosis", {})
        baseline_top = baseline_result.get("top_diagnosis", {})
        multi_name = str(multi_top.get("name", "")).strip()
        baseline_name = str(baseline_top.get("name", "")).strip()
        summary: dict[str, Any] = {
            "multi_agent_top_diagnosis": multi_top,
            "baseline_top_diagnosis": baseline_top,
            "same_top_diagnosis": bool(multi_name and baseline_name and multi_name == baseline_name),
        }
        if baseline_disabled:
            summary["baseline_disabled"] = True
        return summary

    def _save_error_log(
        self,
        error: RealOutputRequiredError,
        problem_name: str,
        case_text: str,
        stage: str,
    ) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "stage": error.stage or stage,
            "agent_name": error.agent_name,
            "provider": error.provider,
            "model": error.model,
            "error_type": error.raw_error_type,
            "error_message": error.reason,
            "request_timeout": self.settings.request_timeout,
            "problem_name": problem_name,
            "case_text_hash": sha256(case_text.encode("utf-8")).hexdigest(),
        }
        self.run_store.save_error_log(payload)

    @staticmethod
    def _collect_report_quality_issues(
        multi_agent_markdown: str,
        baseline_markdown: str,
        baseline_error: dict[str, Any] | None = None,
        baseline_skipped: bool = False,
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        checks = [("multi_agent_markdown", multi_agent_markdown)]
        if baseline_skipped:
            pass
        elif baseline_error:
            issues.append(
                {
                    "source": "baseline_markdown",
                    "code": "BASELINE_REPORT_UNAVAILABLE",
                    "message": str(baseline_error.get("message", "")).strip() or "单模型基线报告不可用",
                }
            )
        else:
            checks.append(("baseline_markdown", baseline_markdown))
        for source, markdown in checks:
            try:
                validate_markdown_report(markdown)
            except ValueError as err:
                issues.append(
                    {
                        "source": source,
                        "code": "REPORT_QUALITY_VALIDATION_FAILED",
                        "message": str(err),
                    }
                )
        return issues
