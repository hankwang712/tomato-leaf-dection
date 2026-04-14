"""将诊断链路中的结构化字段改写为可直接喂给大模型的中文知识叙述（避免 JSON/键名堆砌）。"""

from __future__ import annotations

from typing import Any

from app.core.caption.presentation import localize_caption_payload
from app.core.caption.schema import CaptionSchema

_SYM_LABELS: dict[str, str] = {
    "color": "色泽印象",
    "tissue_state": "组织变化",
    "spot_shape": "病斑形态",
    "boundary": "边界特征",
    "distribution_position": "分布部位",
    "distribution_pattern": "扩展与分布方式",
    "morph_change": "整叶形态",
    "pest_cues": "虫害线索",
    "co_signs": "环境伴随",
}


def caption_to_knowledge_narrative(caption: CaptionSchema) -> str:
    """把 CaptionSchema 转成若干段可读知识，不输出 JSON。"""
    parts: list[str] = [caption.visual_summary.strip()]

    loc = localize_caption_payload(caption.model_dump(mode="json"))
    sym = loc.get("symptoms") if isinstance(loc.get("symptoms"), dict) else {}
    chips: list[str] = []
    for key, label in _SYM_LABELS.items():
        vals = sym.get(key) if isinstance(sym, dict) else None
        if not isinstance(vals, list) or not vals:
            continue
        cn = [str(v).strip() for v in vals[:5] if str(v).strip()]
        if cn:
            chips.append(f"{label}「{'、'.join(cn)}」")
    if chips:
        parts.append("从标签化视觉语义看，还可以这样理解这片叶子：" + "；".join(chips) + "。")

    n = caption.numeric
    parts.append(
        f"量化侧写：病损大约占单叶面积的 {n.area_ratio * 100:.1f}%，"
        f"严重度指数约 {n.severity_score:.2f}（0–1）；"
        f"模型对本帧归类的主观置信约 {caption.confidence * 100:.1f}%。"
    )
    if float(caption.ood_score) >= 0.2:
        parts.append(
            f"图像与训练常见样貌的偏离度（OOD）约 {caption.ood_score:.2f}，"
            "写作时宜保留「可能属于不典型表现」的余地，不要写成铁口直断。"
        )
    fq = [str(x).strip() for x in caption.followup_questions if str(x).strip()]
    if fq:
        parts.append("若田间还能补充信息，优先澄清：" + "；".join(fq[:6]) + "。")
    return "\n\n".join(parts)


def uncertainty_management_to_prose(um: dict[str, Any] | None) -> str:
    if not isinstance(um, dict) or not um:
        return "（当前无不确性与冲突的专门整理；可依据多智能体摘要自行把握审慎语气。）"
    paragraphs: list[str] = []
    cp = um.get("conflict_point")
    if isinstance(cp, dict):
        oi = str(cp.get("overall_impression", "")).strip()
        li = str(cp.get("local_lesion_impression", "")).strip()
        cs = str(cp.get("conflict_summary", "")).strip()
        ms = str(cp.get("model_score_interpretation", "")).strip()
        ec = str(cp.get("evidence_ceiling", "")).strip()
        if oi or li:
            paragraphs.append(
                "整体印象与局部病斑读数之间可能存在张力："
                + (f"整体倾向可概括为「{oi}」。" if oi else "")
                + (f"局部病斑侧写：{li}" if li else "")
            )
        if cs:
            paragraphs.append(f"冲突该如何理解（供你内化，勿照抄）：{cs}")
        if ms:
            paragraphs.append(ms)
        if ec:
            paragraphs.append(f"证据上限（知识层面）：{ec}")

    kds = um.get("key_discriminators")
    if isinstance(kds, list) and kds:
        paragraphs.append(
            "哪些观察最能「撬动」排序——从知识角度应这样看："
        )
        for item in kds[:6]:
            if not isinstance(item, dict):
                continue
            gap = str(item.get("gap", "")).strip()
            dv = str(item.get("diagnostic_value", "")).strip()
            ns = str(item.get("next_step", "")).strip()
            rh = str(item.get("rank_shift_hint", "")).strip()
            if not gap:
                continue
            seg = f"若关心「{gap}」：它对判断的价值在于{dv or '…'}。"
            if ns:
                seg += f" 田间可尝试：{ns}"
            if rh:
                seg += f" {rh}"
            paragraphs.append(seg)
    return "\n\n".join(paragraphs) if paragraphs else "（材料为空。）"


def decision_support_to_prose(ds: dict[str, Any] | None) -> str:
    if not isinstance(ds, dict) or not ds:
        return "（当前无分阶段决策与阈值提示；可依据防治与环境专家摘录写作。）"
    paragraphs: list[str] = []

    cur = ds.get("current_stage_actions")
    if isinstance(cur, list) and cur:
        paragraphs.append(
            "当下阶段在知识上通常优先考虑的田间动作包括："
            + "；".join(str(x).strip() for x in cur[:6] if str(x).strip())
            + "。"
        )

    obs = ds.get("observe_24_48h")
    if isinstance(obs, list) and obs:
        lines: list[str] = []
        for row in obs[:6]:
            if isinstance(row, dict):
                it = str(row.get("item", "")).strip()
                th = str(row.get("threshold_hint", "")).strip()
                if it:
                    lines.append(f"{it}" + (f"（阈值提示：{th}）" if th else ""))
            elif str(row).strip():
                lines.append(str(row).strip())
        if lines:
            paragraphs.append("24–48 小时窗口内值得盯的信号：" + "；".join(lines) + "。")

    up = ds.get("upgrade_conditions")
    if isinstance(up, list) and up:
        paragraphs.append(
            "出现下列情况时，知识上应倾向「提高处理与复核强度」："
            + "；".join(str(x).strip() for x in up[:5] if str(x).strip())
            + "。"
        )
    down = ds.get("downgrade_conditions")
    if isinstance(down, list) and down:
        paragraphs.append(
            "若观察到这些走向，则可考虑维持或下调强度："
            + "；".join(str(x).strip() for x in down[:4] if str(x).strip())
            + "。"
        )

    pro = ds.get("prohibited_actions")
    if isinstance(pro, list) and pro:
        paragraphs.append(
            "边界上不宜做的事："
            + "；".join(str(x).strip() for x in pro[:5] if str(x).strip())
            + "。"
        )

    rev = ds.get("review_nodes")
    if isinstance(rev, list) and rev:
        paragraphs.append(
            "复查与时间节点上的知识提醒："
            + "；".join(str(x).strip() for x in rev[:5] if str(x).strip())
            + "。"
        )

    br = ds.get("post_review_branches")
    if isinstance(br, list) and br:
        paragraphs.append(
            "补证之后常见的两条叙事分支："
            + " ".join(str(x).strip() for x in br[:4] if str(x).strip())
        )

    return "\n\n".join(paragraphs) if paragraphs else "（材料为空。）"


def _bundle_visual_only(b: dict[str, Any]) -> str:
    vs = str(b.get("visual_summary", "")).strip()
    morph = b.get("morphology_cues")
    ext = str(b.get("extent_note", "")).strip()
    co = str(b.get("conflict_one_liner", "")).strip()
    wn = str(b.get("writing_note", "")).strip()
    segs = []
    if vs:
        segs.append(vs)
    if isinstance(morph, list) and morph:
        segs.append("形态线索可记为：" + "、".join(str(x) for x in morph[:6] if str(x).strip()) + "。")
    if ext:
        segs.append(ext)
    if co:
        segs.append(f"与模型整体印象相关的张力：{co}")
    if wn:
        segs.append(wn)
    return "\n".join(segs) if segs else ""


def section_facts_to_knowledge_narrative(facts: Any, section_title: str) -> str:
    """按章节把 section_facts 翻成知识段落，供本章撰稿专用。"""
    if not isinstance(facts, dict) or not facts:
        return "（本节没有单独附带的要点包。）"

    st = str(section_title or "").strip()
    out: list[str] = []

    if st == "基本信息":
        crop = str(facts.get("crop", "番茄")).strip()
        pd = str(facts.get("primary_diagnosis", "")).strip()
        cl = str(facts.get("confidence_label", "")).strip()
        msn = str(facts.get("model_score_note", "")).strip()
        sh = str(facts.get("stage_hint", "")).strip()
        cc = facts.get("classification_confidence", 0.0)
        sd = str(facts.get("secondary_differential", "")).strip()
        vs = str(facts.get("visual_summary", "")).strip()
        if crop:
            out.append(f"作物：{crop}")
        if pd:
            out.append(f"首位疑似诊断：{pd}")
        if cl:
            out.append(f"置信层级：{cl}")
        if msn:
            out.append(f"模型分数说明：{msn}")
        if sh:
            out.append(f"病害阶段：{sh}")
        if sd:
            out.append(f"主要鉴别对象：{sd}")
        if vs:
            out.append(f"视觉概述：{vs}")
        snippets = facts.get("disease_context_snippets")
        if isinstance(snippets, list) and snippets:
            out.append("知识库摘录：" + " ".join(_clip(str(s), 400) for s in snippets[:5] if str(s).strip()))

    elif st == "症状观察":
        morph = facts.get("morphology")
        if isinstance(morph, list) and morph:
            out.append("可见形态特征：" + "、".join(str(x) for x in morph[:8] if str(x).strip()))
        vs = str(facts.get("visual_summary", "")).strip()
        if vs:
            out.append(f"视觉摘要：{vs}")
        vb = facts.get("visual_only_bundle")
        if isinstance(vb, dict):
            prose = _bundle_visual_only(vb)
            if prose:
                out.append(prose)
        cn = str(facts.get("consistency_note", "")).strip()
        if cn:
            out.append(cn)
        de = facts.get("diagnosis_evidence")
        if isinstance(de, list) and de:
            for entry in de[:3]:
                if isinstance(entry, dict):
                    d = str(entry.get("diagnosis", "")).strip()
                    sup = entry.get("supporting", [])
                    if d and isinstance(sup, list) and sup:
                        out.append(f"{d}的支持证据：" + "；".join(str(x) for x in sup[:3] if str(x).strip()))

    elif st == "鉴别诊断":
        pd = str(facts.get("primary_diagnosis", "")).strip()
        sd = str(facts.get("secondary_differential", "")).strip()
        if pd:
            out.append(f"首位疑似：{pd}")
        if sd:
            out.append(f"主要鉴别对象：{sd}")
        dst = str(facts.get("diagnosis_statement", "")).strip()
        if dst:
            out.append(f"诊断概括：{dst}")
        cl = str(facts.get("confidence_label", "")).strip()
        msn = str(facts.get("model_score_note", "")).strip()
        if cl or msn:
            out.append(f"置信说明：{cl or ''} {msn or ''}".strip())
        snippets = facts.get("disease_context_snippets")
        if isinstance(snippets, list) and snippets:
            out.append("知识库摘录：" + " ".join(_clip(str(s), 400) for s in snippets[:5] if str(s).strip()))
        db = facts.get("diagnosis_board")
        if isinstance(db, dict):
            diffs = db.get("differentials")
            if isinstance(diffs, list) and diffs:
                for d in diffs[:4]:
                    if not isinstance(d, dict):
                        continue
                    nm = str(d.get("name", "")).strip()
                    ws = str(d.get("why_supported", "")).strip()
                    wn = str(d.get("why_not_primary", "")).strip()
                    if nm:
                        out.append(f"候选[{nm}]: 支持—{ws or chr(8230)}；限制—{wn or chr(8230)}")
        vc = facts.get("vision_conflict")
        if isinstance(vc, dict) and vc.get("has_conflict"):
            rs = str(vc.get("reason_summary", "")).strip()
            if rs:
                out.append(f"视觉分歧：{rs}")

    elif st == "防治建议":
        ta = facts.get("today_actions")
        if isinstance(ta, list) and ta:
            out.append("立即可执行：" + "；".join(str(x) for x in ta[:6] if str(x).strip()))
        co = facts.get("control_options")
        if isinstance(co, list) and co:
            out.append("防治方案：" + "；".join(str(x) for x in co[:5] if str(x).strip()))
        o48 = facts.get("observe_48h")
        if isinstance(o48, list) and o48:
            out.append("近两日观察重点：" + "；".join(str(x) for x in o48[:5] if str(x).strip()))
        esc = facts.get("escalation_conditions")
        if isinstance(esc, list) and esc:
            out.append("升级条件：" + "；".join(str(x) for x in esc[:4] if str(x).strip()))
        gis = facts.get("gap_items")
        if isinstance(gis, list) and gis:
            out.append("需核查的证据：")
            for g in gis[:4]:
                if isinstance(g, dict):
                    gap = str(g.get("gap", "")).strip()
                    ns = str(g.get("next_step", "")).strip()
                    if gap:
                        out.append(f"- {gap}：{ns}" if ns else f"- {gap}")

    elif st == "预后评估":
        pa = facts.get("prohibited_actions")
        if isinstance(pa, list) and pa:
            out.append("禁忌事项：" + "；".join(str(x) for x in pa[:6] if str(x).strip()))
        sn = facts.get("safety_notes")
        if isinstance(sn, list) and sn:
            out.append("风险信号：" + "；".join(str(x) for x in sn[:6] if str(x).strip()))
        rf = facts.get("required_followups")
        if isinstance(rf, list) and rf:
            out.append("复查节点：" + "；".join(str(x) for x in rf[:6] if str(x).strip()))
        pn = str(facts.get("prognosis_note", "")).strip()
        if pn:
            out.append(f"预后：{pn}")
        ta = facts.get("today_actions")
        if isinstance(ta, list) and ta:
            out.append("当前措施：" + "；".join(str(x) for x in ta[:4] if str(x).strip()))

    elif st == "备注":
        es = str(facts.get("evidence_sufficiency", "")).strip()
        if es:
            out.append(f"证据边界：{es}")
        pd = str(facts.get("primary_diagnosis", "")).strip()
        sd = str(facts.get("secondary_differential", "")).strip()
        if pd or sd:
            out.append("诊断站位：" + " / ".join(x for x in (pd, sd) if x))
        cb = str(facts.get("conflict_brief", "")).strip()
        if cb:
            out.append(f"关键张力：{cb}")
        rf = facts.get("required_followups")
        if isinstance(rf, list) and rf:
            out.append("优先跟进：" + "；".join(str(x) for x in rf[:4] if str(x).strip()))
        um = facts.get("uncertainty_management")
        if isinstance(um, dict) and um:
            out.append(uncertainty_management_to_prose(um))

    else:
        for k, v in list(facts.items())[:15]:
            if isinstance(v, str) and v.strip():
                out.append(f"{k}：{v.strip()}")
            elif isinstance(v, list) and v:
                out.append(f"{k}：" + "；".join(str(x).strip() for x in v[:8] if str(x).strip()))

    text = "\n\n".join(x for x in out if str(x).strip())
    return text if len(text) > 20 else text + "\n\n（请结合全局材料展开。）"


def _clip(s: str, n: int) -> str:
    t = str(s).strip()
    return t if len(t) <= n else t[: n - 1] + "…"
