"""信号转导层 (Signal Transduction Layer)

植物免疫启发架构中的信号转导模块。模拟植物 MAPK 级联信号通路：
将视觉模式识别层（PRR Layer）的结构化输出转化为语义级机制标签，
并通过 LLM 语义理解（而非关键词匹配）驱动证据源路由决策。

取代旧版基于硬编码关键词与阈值的路由方式，
所有语义判断均由 LLM 在上下文中完成，消除关键词漏检的不确定性。
"""
from __future__ import annotations

import json
from typing import Any

from app.core.caption.schema import (
    CaptionSchema,
    CoSignEnum,
    DistributionPositionEnum,
    PestCueEnum,
    TissueStateEnum,
)


# ---------------------------------------------------------------------------
# 结构化字段快捷提取（零 LLM 开销，基于 CaptionSchema 枚举）
# ---------------------------------------------------------------------------

def _enum_values(items: list) -> list[str]:
    return [str(getattr(item, "value", item)).strip() for item in items if str(getattr(item, "value", item)).strip()]


def _plant_part(caption: CaptionSchema) -> str:
    positions = {item.value for item in caption.symptoms.distribution_position}
    mapping = [
        (DistributionPositionEnum.fruit.value, "fruit"),
        (DistributionPositionEnum.stem.value, "stem"),
        (DistributionPositionEnum.whole_plant.value, "whole_plant"),
    ]
    for enum_val, part_name in mapping:
        if enum_val in positions:
            return part_name
    return sorted(positions)[0] if positions else "leaf"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


# ---------------------------------------------------------------------------
# LLM 驱动的语义标签提取
# ---------------------------------------------------------------------------

_MECHANISM_EXTRACTION_PROMPT = """你是一个植物病理学信号分析模块。你的任务是从给定的视觉观察和田间描述中，
提取结构化的诊断机制标签。不要编造，只基于给定信息推断。

输入信息：
{context}

请返回严格 JSON，包含以下字段：
{{
  "growth_stage": "seedling|flowering|fruiting|mature|unknown",
  "environment_signals": ["humidity_high", "poor_ventilation", "overwatering", "rainy_weather", "dew_presence", ...] 或空列表,
  "stressor_class": ["fungal", "bacterial", "viral", "pest", "phytotoxicity", "nutritional", "environmental_stress", ...] 或空列表,
  "progression_hints": ["initial", "expanding", "severe", "needs_recheck", ...] 或空列表,
  "reasoning": "一句话解释你的判断依据"
}}

规则：
- growth_stage 从田间描述推断；无线索时填 unknown
- environment_signals 从环境描述和伴随征象推断
- stressor_class 根据组织状态、形态特征推断可能的胁迫来源类别
- progression_hints 根据受害面积、严重度、分布模式推断
- 只输出 JSON，不要输出其他内容"""


def build_mechanism_extraction_context(
    caption: CaptionSchema,
    *,
    case_text: str = "",
    image_display: dict[str, Any] | None = None,
    vision_result: dict[str, Any] | None = None,
) -> str:
    """将所有可用信息组装为 LLM 可理解的自然语言上下文。"""
    parts = []

    parts.append(f"【视觉摘要】{caption.visual_summary}")

    symptom = caption.symptoms
    parts.append(f"【颜色】{', '.join(_enum_values(symptom.color))}")
    parts.append(f"【组织状态】{', '.join(_enum_values(symptom.tissue_state))}")
    parts.append(f"【斑点形状】{', '.join(_enum_values(symptom.spot_shape))}")
    parts.append(f"【边界特征】{', '.join(_enum_values(symptom.boundary))}")
    parts.append(f"【分布位置】{', '.join(_enum_values(symptom.distribution_position))}")
    parts.append(f"【分布模式】{', '.join(_enum_values(symptom.distribution_pattern))}")
    parts.append(f"【形态变化】{', '.join(_enum_values(symptom.morph_change))}")
    parts.append(f"【虫害线索】{', '.join(_enum_values(symptom.pest_cues))}")
    parts.append(f"【伴随征象】{', '.join(_enum_values(symptom.co_signs))}")

    parts.append(f"【受害面积比】{caption.numeric.area_ratio:.2f}")
    parts.append(f"【严重度评分】{caption.numeric.severity_score:.2f}")
    parts.append(f"【分类置信度】{caption.confidence:.2f}")
    parts.append(f"【分布外评分】{caption.ood_score:.2f}")

    if caption.followup_questions:
        parts.append(f"【待补证项】{'; '.join(caption.followup_questions[:5])}")

    if case_text:
        parts.append(f"【田间描述】{case_text[:500]}")

    if isinstance(image_display, dict):
        summary = str(image_display.get("摘要", "")).strip()
        if summary:
            parts.append(f"【图像分析摘要】{summary}")

    if isinstance(vision_result, dict):
        conflict = vision_result.get("conflict_analysis", {})
        if isinstance(conflict, dict) and conflict.get("reason_summary"):
            parts.append(f"【视觉冲突分析】{conflict['reason_summary']}")

    return "\n".join(parts)


def extract_mechanism_tags_via_llm(
    llm_client: Any,
    caption: CaptionSchema,
    *,
    case_text: str = "",
    image_display: dict[str, Any] | None = None,
    vision_result: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """通过 LLM 语义理解提取机制标签，取代旧版关键词匹配。"""
    context = build_mechanism_extraction_context(
        caption, case_text=case_text, image_display=image_display, vision_result=vision_result,
    )
    prompt = _MECHANISM_EXTRACTION_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": "你是农业病害诊断系统的信号转导模块。只返回严格 JSON。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.generate(messages=messages, temperature=0.1, timeout=timeout)
        text = str(response.content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
    except Exception:
        parsed = {}

    return _normalize_mechanism_tags(parsed, caption, case_text)


def _normalize_mechanism_tags(
    parsed: dict[str, Any],
    caption: CaptionSchema,
    case_text: str,
) -> dict[str, Any]:
    """规范化 LLM 输出，确保字段完整且类型正确；同时保留结构化字段的兜底推断。"""
    growth_stage = str(parsed.get("growth_stage", "unknown")).strip() or "unknown"

    environment_signals = _ordered_unique(
        [str(s).strip() for s in parsed.get("environment_signals", []) if str(s).strip()]
    )
    co_signs = _enum_values(caption.symptoms.co_signs)
    for sign in co_signs:
        if sign != "unknown" and sign not in environment_signals:
            environment_signals.append(sign)

    stressor_class = _ordered_unique(
        [str(s).strip() for s in parsed.get("stressor_class", []) if str(s).strip()]
    )
    if not stressor_class:
        stressor_class = _fallback_stressor_class(caption)

    progression_hints = _ordered_unique(
        [str(s).strip() for s in parsed.get("progression_hints", []) if str(s).strip()]
    )
    if not progression_hints:
        progression_hints = _fallback_progression_hints(caption)

    return {
        "host_context": {
            "crop": "tomato",
            "plant_part": _plant_part(caption),
            "growth_stage": growth_stage,
        },
        "environment_signals": environment_signals or ["unknown"],
        "stressor_class": stressor_class or ["unknown"],
        "progression_hints": progression_hints or ["unknown"],
        "llm_reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def _fallback_stressor_class(caption: CaptionSchema) -> list[str]:
    """当 LLM 未能返回有效结果时，基于结构化枚举字段做最低限度推断。"""
    classes: list[str] = []
    tissue_states = {item.value for item in caption.symptoms.tissue_state}
    pest_cues = {item.value for item in caption.symptoms.pest_cues}

    if TissueStateEnum.mold.value in tissue_states:
        classes.append("fungal")
    if TissueStateEnum.water_soaked.value in tissue_states:
        classes.append("bacterial")
    if pest_cues - {PestCueEnum.no_obvious_pest.value}:
        classes.append("pest")
    return classes or ["unknown"]


def _fallback_progression_hints(caption: CaptionSchema) -> list[str]:
    """基于数值字段做最低限度推断。"""
    hints: list[str] = []
    if caption.numeric.severity_score >= 0.7 or caption.numeric.area_ratio >= 0.4:
        hints.append("severe")
    elif caption.numeric.severity_score <= 0.25 and caption.numeric.area_ratio <= 0.12:
        hints.append("initial")
    if caption.followup_questions or caption.ood_score >= 0.15:
        hints.append("needs_recheck")
    return hints or ["unknown"]


# ---------------------------------------------------------------------------
# 兼容旧接口：无 LLM 时的纯结构化提取（降级路径）
# ---------------------------------------------------------------------------

def extract_mechanism_tags(
    caption: CaptionSchema,
    *,
    case_text: str = "",
    image_display: dict[str, Any] | None = None,
    shared_state: dict[str, Any] | None = None,
    vision_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """结构化字段提取（无 LLM 调用），作为 LLM 提取的降级兜底。

    当 pipeline 层传入 llm_client 时应优先使用 extract_mechanism_tags_via_llm。
    """
    co_signs = _enum_values(caption.symptoms.co_signs)
    environment_signals = [s for s in co_signs if s != "unknown"] or ["unknown"]

    return {
        "host_context": {
            "crop": "tomato",
            "plant_part": _plant_part(caption),
            "growth_stage": "unknown",
        },
        "environment_signals": environment_signals,
        "stressor_class": _fallback_stressor_class(caption),
        "progression_hints": _fallback_progression_hints(caption),
    }


# ---------------------------------------------------------------------------
# LLM 驱动的源路由决策
# ---------------------------------------------------------------------------

_ROUTING_DECISION_PROMPT = """你是植物病害诊断系统的信号路由模块。
根据以下诊断信号，决定证据检索策略。

诊断信号：
- 分类置信度: {confidence:.2f}（越高越确定）
- 分布外评分: {ood_score:.2f}（越高越不常见）
- 历史病例匹配度: {case_support:.2f}（越高说明历史库中有类似案例）
- 待补证缺口数: {gap_count}
- 胁迫类别: {stressor_class}
- 环境信号: {environment_signals}
- 病程阶段: {progression_hints}

请返回严格 JSON：
{{
  "mode": "case_priority|document_priority|mixed",
  "reasoning": "一句话解释路由决策",
  "verified_k": 3到10之间的整数,
  "unverified_k": 2到6之间的整数,
  "document_k": 3到10之间的整数
}}

决策原则：
- case_priority: 当历史匹配度高且视觉证据清晰时，优先参考历史病例
- document_priority: 当证据不确定、OOD高、缺口多时，优先查阅知识文档
- mixed: 证据信号混合时，均衡检索
- verified_k + unverified_k + document_k 的总预算约 16 条
- 只输出 JSON"""


class SourceRouter:
    """LLM 驱动的证据源路由器。

    取代旧版基于硬编码阈值的路由逻辑，
    通过 LLM 语义理解来判断应优先检索哪类证据源。
    """

    def __init__(self, *, llm_client: Any = None, timeout: int = 30):
        self.llm_client = llm_client
        self.timeout = timeout

    def route(
        self,
        *,
        caption: CaptionSchema,
        mechanism_tags: dict[str, Any],
        case_support: float,
        shared_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.llm_client is not None:
            return self._route_via_llm(
                caption=caption,
                mechanism_tags=mechanism_tags,
                case_support=case_support,
                shared_state=shared_state,
            )
        return self._route_heuristic(
            caption=caption,
            mechanism_tags=mechanism_tags,
            case_support=case_support,
            shared_state=shared_state,
        )

    def _route_via_llm(
        self,
        *,
        caption: CaptionSchema,
        mechanism_tags: dict[str, Any],
        case_support: float,
        shared_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gap_count = len((shared_state or {}).get("evidence_gaps", []) or caption.followup_questions)
        prompt = _ROUTING_DECISION_PROMPT.format(
            confidence=caption.confidence,
            ood_score=caption.ood_score,
            case_support=case_support,
            gap_count=gap_count,
            stressor_class=", ".join(mechanism_tags.get("stressor_class", ["unknown"])),
            environment_signals=", ".join(mechanism_tags.get("environment_signals", ["unknown"])),
            progression_hints=", ".join(mechanism_tags.get("progression_hints", ["unknown"])),
        )
        messages = [
            {"role": "system", "content": "你是诊断信号路由模块。只返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.llm_client.generate(messages=messages, temperature=0.1, timeout=self.timeout)
            text = str(response.content).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
        except Exception:
            return self._route_heuristic(
                caption=caption, mechanism_tags=mechanism_tags,
                case_support=case_support, shared_state=shared_state,
            )

        return self._normalize_routing(parsed, case_support, "llm")

    def _route_heuristic(
        self,
        *,
        caption: CaptionSchema,
        mechanism_tags: dict[str, Any],
        case_support: float,
        shared_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """当 LLM 不可用时的纯数值启发式降级。"""
        low_ood = caption.ood_score <= 0.18
        strong_visual = caption.confidence >= 0.72

        if low_ood and strong_visual and case_support >= 0.38:
            return self._build_result("case_priority", case_support, "heuristic_fallback",
                                      verified_k=8, unverified_k=4, document_k=3)
        gap_count = len((shared_state or {}).get("evidence_gaps", []) or caption.followup_questions)
        if caption.ood_score >= 0.26 or caption.confidence <= 0.55 or case_support < 0.18 or gap_count >= 3:
            return self._build_result("document_priority", case_support, "heuristic_fallback",
                                      verified_k=3, unverified_k=2, document_k=8)
        return self._build_result("mixed", case_support, "heuristic_fallback",
                                  verified_k=6, unverified_k=4, document_k=5)

    def _normalize_routing(self, parsed: dict[str, Any], case_support: float, source: str) -> dict[str, Any]:
        mode = str(parsed.get("mode", "mixed")).strip()
        if mode not in ("case_priority", "document_priority", "mixed"):
            mode = "mixed"
        verified_k = max(1, min(10, int(parsed.get("verified_k", 6) or 6)))
        unverified_k = max(1, min(6, int(parsed.get("unverified_k", 4) or 4)))
        document_k = max(1, min(10, int(parsed.get("document_k", 5) or 5)))
        return self._build_result(
            mode, case_support, source,
            verified_k=verified_k, unverified_k=unverified_k, document_k=document_k,
            reasoning=str(parsed.get("reasoning", "")).strip(),
        )

    @staticmethod
    def _build_result(
        mode: str, case_support: float, source: str, *,
        verified_k: int, unverified_k: int, document_k: int,
        reasoning: str = "",
    ) -> dict[str, Any]:
        weight_map = {
            "case_priority": (1.25, 0.85),
            "document_priority": (0.9, 1.25),
            "mixed": (1.08, 1.08),
        }
        case_w, doc_w = weight_map.get(mode, (1.0, 1.0))
        return {
            "mode": mode,
            "case_weight": case_w,
            "document_weight": doc_w,
            "mixed_mode": mode == "mixed",
            "routing_reason": reasoning or f"{source}_{mode}",
            "budgets": {"verified_k": verified_k, "unverified_k": unverified_k, "document_k": document_k},
            "case_support": round(float(case_support), 4),
        }


# ---------------------------------------------------------------------------
# 源一致性分析（保持兼容）
# ---------------------------------------------------------------------------

def summarize_source_alignment(
    *,
    caption: CaptionSchema,
    mechanism_tags: dict[str, Any],
    source_routing: dict[str, Any],
    case_records: list[dict[str, Any]],
    document_records: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    agreements: list[str] = []
    conflicts: list[str] = []

    case_support = float(source_routing.get("case_support", 0.0) or 0.0)
    if case_records and case_support >= 0.38:
        agreements.append("历史病例与当前症状机制标签存在较高匹配。")
    if document_records:
        agreements.append("外部文档可为当前病例提供规则与风险边界补充。")
    if caption.ood_score >= 0.25:
        conflicts.append("图像分布外程度较高，不能仅依赖单一来源快速收敛。")
    if source_routing.get("mode") == "mixed":
        conflicts.append("当前病例经验与文档规则需联合解释，暂不宜单源定论。")
    if case_support < 0.18 and document_records:
        conflicts.append("历史病例支持较弱，文档证据应占更高权重。")
    stressor_class = set(str(item) for item in mechanism_tags.get("stressor_class", []))
    if len(stressor_class - {"unknown"}) >= 2:
        conflicts.append("多种胁迫类别并存，需综合多源证据交叉验证。")

    return _ordered_unique(agreements), _ordered_unique(conflicts)


def build_mechanism_hypotheses(mechanism_tags: dict[str, Any]) -> list[str]:
    hypotheses: list[str] = []
    environment_signals = set(str(item) for item in mechanism_tags.get("environment_signals", []) if str(item).strip())
    stressor_class = set(str(item) for item in mechanism_tags.get("stressor_class", []) if str(item).strip())
    progression_hints = set(str(item) for item in mechanism_tags.get("progression_hints", []) if str(item).strip())

    if "fungal" in stressor_class and {"humidity_high", "poor_ventilation", "rainy_weather", "dew_presence"} & environment_signals:
        hypotheses.append("高湿与霉变相关线索共同支持真菌或卵菌性叶部病害优先排查。")
    if "bacterial" in stressor_class:
        hypotheses.append("水渍样或黄晕类线索提示需保留细菌性病害鉴别。")
    if "viral" in stressor_class:
        hypotheses.append("花叶、畸形或系统性症状提示病毒侵染可能，需观察全株表现。")
    if "pest" in stressor_class:
        hypotheses.append("虫孔或虫害线索存在时，应避免把全部损伤都解释为病原侵染。")
    if {"environmental_stress", "nutritional", "phytotoxicity"} & stressor_class:
        hypotheses.append("当前表现可能包含非病原性胁迫，需结合环境管理与处理史解释。")
    if "severe" in progression_hints:
        hypotheses.append("受害范围已偏重，后续动作应更保守并保留升级复核节点。")
    if "needs_recheck" in progression_hints:
        hypotheses.append("现有证据仍需复拍或补证，不宜仅凭单源信号提前收敛。")
    return _ordered_unique(hypotheses)


# ---------------------------------------------------------------------------
# 机制重叠度计算（用于病例检索排序）
# ---------------------------------------------------------------------------

def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mechanism_overlap_score(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0

    left_host = left.get("host_context", {}) if isinstance(left.get("host_context"), dict) else {}
    right_host = right.get("host_context", {}) if isinstance(right.get("host_context"), dict) else {}

    crop_score = 1.0 if (left_host.get("crop") and left_host.get("crop") == right_host.get("crop")) else 0.0
    part_score = 1.0 if (left_host.get("plant_part") and left_host.get("plant_part") == right_host.get("plant_part")) else 0.0

    left_stage = str(left_host.get("growth_stage", "")).strip()
    right_stage = str(right_host.get("growth_stage", "")).strip()
    stage_score = 1.0 if (left_stage and right_stage and left_stage != "unknown" and right_stage != "unknown" and left_stage == right_stage) else 0.0

    environment_score = _jaccard(
        {str(item) for item in left.get("environment_signals", []) if str(item).strip() and str(item) != "unknown"},
        {str(item) for item in right.get("environment_signals", []) if str(item).strip() and str(item) != "unknown"},
    )
    stressor_score = _jaccard(
        {str(item) for item in left.get("stressor_class", []) if str(item).strip() and str(item) != "unknown"},
        {str(item) for item in right.get("stressor_class", []) if str(item).strip() and str(item) != "unknown"},
    )
    progression_score = _jaccard(
        {str(item) for item in left.get("progression_hints", []) if str(item).strip() and str(item) != "unknown"},
        {str(item) for item in right.get("progression_hints", []) if str(item).strip() and str(item) != "unknown"},
    )

    weighted = (
        crop_score * 0.2
        + part_score * 0.15
        + stage_score * 0.1
        + environment_score * 0.2
        + stressor_score * 0.2
        + progression_score * 0.15
    )
    return max(0.0, min(1.0, weighted))
