from __future__ import annotations

from typing import Any


def normalize_label(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


_CLASS_NAME_CN = {
    "leaf": "叶片",
    "bacterialspot": "细菌性斑点病",
    "tomatobacterialspot": "细菌性斑点病",
    "earlyblight": "早疫病",
    "tomatoearlyblight": "早疫病",
    "healthy": "健康",
    "tomatohealthy": "健康",
    "lateblight": "晚疫病",
    "tomatolateblight": "晚疫病",
    "leafmold": "叶霉病",
    "tomatoleafmold": "叶霉病",
    "septorialeafspot": "白星病",
    "tomatoseptorialeafspot": "白星病",
    "spidermitestwospottedspidermite": "二斑叶螨",
    "tomatospidermitestwospottedspidermite": "二斑叶螨",
    "targetspot": "靶斑病",
    "tomatotargetspot": "靶斑病",
    "tomatomosaicvirus": "番茄花叶病毒病",
    "tomatotomatomosaicvirus": "番茄花叶病毒病",
    "tomatoyellowleafcurlvirus": "番茄黄化曲叶病毒病",
    "tomatotomatoyellowleafcurlvirus": "番茄黄化曲叶病毒病",
}


def class_name_to_cn(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "未知"
    mapped = _CLASS_NAME_CN.get(normalize_label(text))
    if mapped:
        return mapped
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    return f"未映射类别（{text}）"


def format_percent(value: float) -> str:
    return f"{clamp_unit(value) * 100:.1f}%"


def build_conflict_analysis(image_analysis: dict[str, Any]) -> dict[str, Any]:
    predicted_class = str(image_analysis.get("predicted_class", "")).strip()
    predicted_norm = normalize_label(predicted_class)
    confidence = clamp_unit(float(image_analysis.get("confidence", 0.0)))
    damage_ratio = clamp_unit(float(image_analysis.get("damaged_area_ratio_of_leaf", 0.0)))
    dominant_ratio = clamp_unit(float(image_analysis.get("dominant_segmentation_ratio_of_leaf", 0.0)))
    predicted_damage_ratio = clamp_unit(float(image_analysis.get("predicted_class_damage_ratio_of_leaf", 0.0)))

    has_conflict = False
    reasons: list[str] = []
    if predicted_norm == "healthy" and damage_ratio >= 0.08:
        has_conflict = True
        reasons.append("分类头输出健康，但受损面积已达到可见水平。")
    if predicted_norm not in {"", "healthy"} and confidence < 0.55 and damage_ratio >= 0.35:
        has_conflict = True
        reasons.append("分类头置信度偏低且受损面积较大，病种结论需要复核。")
    if predicted_damage_ratio > 0 and damage_ratio - predicted_damage_ratio >= 0.15:
        has_conflict = True
        reasons.append("分类相关区域与总受损面积差距较大，需结合时序复拍确认。")
    if not reasons:
        reasons.append("分类病种判断与受损面积信息未见明显矛盾。")

    if has_conflict:
        summary = (
            f"分类结果为“{class_name_to_cn(predicted_class)}”，但病损面积约占叶片 {format_percent(damage_ratio)}，"
            "当前应先补证再提高病种结论强度。"
        )
        recommended_interpretation = "病种判断以分类头为主，分割头仅用于受损面积评估与复查趋势管理。"
    else:
        summary = "分类病种判断与受损面积信息整体一致。"
        recommended_interpretation = "保持当前分类病种判断，同时持续跟踪病损面积变化。"

    return {
        "has_conflict": has_conflict,
        "classification_result": predicted_class,
        "classification_result_cn": class_name_to_cn(predicted_class),
        "classification_confidence": confidence,
        "segmentation_result": "",
        "segmentation_result_cn": "",
        "damaged_area_ratio_of_leaf": damage_ratio,
        "dominant_segmentation_ratio_of_leaf": dominant_ratio,
        "predicted_class_damage_ratio_of_leaf": predicted_damage_ratio,
        "reason_summary": summary,
        "reason_details": reasons,
        "recommended_interpretation": recommended_interpretation,
        "index_alignment_note": "分割头不参与病种判读，仅用于病损面积估计。",
    }


def resolve_primary_visual_diagnosis(image_analysis: dict[str, Any]) -> dict[str, Any]:
    predicted_class = str(image_analysis.get("predicted_class", "")).strip() or "unknown"
    damage_ratio = clamp_unit(float(image_analysis.get("damaged_area_ratio_of_leaf", 0.0)))
    dominant_ratio = clamp_unit(float(image_analysis.get("dominant_segmentation_ratio_of_leaf", 0.0)))
    conflict = build_conflict_analysis(image_analysis)

    consistency_note = "病种判断以分类结果为主，分割结果仅用于估计病斑面积和受损程度。"
    if conflict["has_conflict"]:
        consistency_note = "当前病种判断仍以分类结果为主，但面积信号提示需在 24 到 48 小时内复查。"

    return {
        "primary_class": predicted_class,
        "primary_class_cn": class_name_to_cn(predicted_class),
        "primary_source": "分类头",
        "predicted_class_cn": class_name_to_cn(predicted_class),
        "dominant_segmentation_class_cn": "",
        "has_conflict": conflict["has_conflict"],
        "consistency_note": consistency_note,
        "damage_ratio": damage_ratio,
        "dominant_ratio": dominant_ratio,
        "conflict_analysis": conflict,
    }


def build_image_analysis_display(image_analysis: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_primary_visual_diagnosis(image_analysis)
    conflict = resolved["conflict_analysis"]
    confidence = clamp_unit(float(image_analysis.get("confidence", 0.0)))
    damage_ratio = clamp_unit(float(image_analysis.get("damaged_area_ratio_of_leaf", 0.0)))
    leaf_pixels = int(image_analysis.get("leaf_pixels", 0) or 0)
    diseased_pixels = int(image_analysis.get("diseased_pixels", 0) or 0)

    top_predictions = [
        {
            "名称": class_name_to_cn(str(item.get("class_name", ""))),
            "置信度": format_percent(float(item.get("confidence", 0.0))),
        }
        for item in image_analysis.get("top_predictions", [])
        if isinstance(item, dict)
    ]

    segmentation_findings: list[dict[str, Any]] = []
    for idx, item in enumerate(image_analysis.get("disease_area_details", []), start=1):
        if not isinstance(item, dict):
            continue
        ratio = clamp_unit(float(item.get("ratio_of_leaf", 0.0)))
        if ratio <= 0.0:
            continue
        segmentation_findings.append(
            {
                "区域": f"病斑区域{idx}",
                "病损面积占叶片比例": format_percent(ratio),
                "病损像素": int(item.get("pixels", 0) or 0),
                "平均概率": format_percent(float(item.get("mean_probability", 0.0))),
            }
        )

    summary_text = (
        f"当前图像病种判断更偏向“{resolved['primary_class_cn']}”，主要依据来自分类头；"
        f"分类置信度为 {format_percent(confidence)}，病损面积约占叶片面积 {format_percent(damage_ratio)}。"
        "分割结果仅用于面积评估，不参与病种类型判读。"
    )
    if conflict["has_conflict"]:
        summary_text += " 当前病种与面积信号存在不一致，建议在 24 到 48 小时内复拍复核。"

    return {
        "摘要": summary_text,
        "结论卡片": {
            "综合判断": resolved["primary_class_cn"],
            "综合判断来源": resolved["primary_source"],
            "分类结果": resolved["predicted_class_cn"],
            "分类置信度": format_percent(confidence),
            "分割用途": "仅用于受损面积估计",
            "病损面积占叶片比例": format_percent(damage_ratio),
            "叶片像素": leaf_pixels,
            "病损像素": diseased_pixels,
            "结果一致性": "面积信号需复核" if conflict["has_conflict"] else "病种与面积基本一致",
        },
        "一致性说明": resolved["consistency_note"],
        "结果差异分析": {
            "差异摘要": conflict["reason_summary"],
            "差异原因": conflict["reason_details"],
            "解释建议": conflict["recommended_interpretation"],
            "索引说明": conflict["index_alignment_note"],
        },
        "分类候选": top_predictions,
        "病斑面积明细": segmentation_findings,
    }
