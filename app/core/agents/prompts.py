from __future__ import annotations

import base64
import json
from typing import Any

from app.core.agents.knowledge_prose import caption_to_knowledge_narrative
from app.core.caption.schema import CaptionSchema


REQUIRED_REPORT_SECTIONS = [
    "基本信息",
    "症状观察",
    "鉴别诊断",
    "防治建议",
    "预后评估",
    "备注",
]

# 在专家/协调/报告等 system 首段注入；亦可供 orchestrator 内联提示复用。
MODEL_CAPABILITY_PREAMBLE = (
    "【模型与领域边界】本对话由中小参数语言模型执行；通用训练知识在农艺/植保上不完整且可能过时。"
    "你只被信任为：在**给定输入**（病例文字、视觉摘要结构化字段、知识库/病例摘录、共享状态）之内做推理与整理。"
    "凡输入未逐字给出的药剂通用名、登记含量、稀释倍数、施药间隔等，一律不得编造；"
    "若知识库无相关内容，须明确写出依据缺口，并退回「原则级/类别级」建议外加本地复核路径。"
)


def _caption_payload(caption: CaptionSchema) -> dict[str, Any]:
    return caption.model_dump(mode="json")


def _format_kb_evidence_for_expert(kb_evidence: list[dict[str, Any]], agent_name: str) -> str:
    if not kb_evidence:
        return "（暂无知识库证据）"
    lines = ["=== 知识库证据（请务必引用其中具体内容）===\n"]
    for idx, item in enumerate(kb_evidence, 1):
        source = str(item.get("source", "")).strip()
        if source == "知识库" or item.get("content"):
            title = item.get("title", "")
            content = item.get("content", "")
            if content:
                lines.append(f"【知识库-{idx}】{title}\n{content}\n")
        else:
            diagnosis = item.get("diagnosis", "")
            summary = item.get("summary", "")
            symptoms = item.get("symptoms", "")
            case_line = f"【病例-{idx}】"
            if diagnosis:
                case_line += f"诊断:{diagnosis}；"
            if summary:
                case_line += f"摘要:{summary}；"
            if symptoms:
                symptoms_str = symptoms if isinstance(symptoms, str) else str(symptoms)
                case_line += f"症状:{symptoms_str[:100]}；"
            lines.append(case_line.strip() + "\n")
    return "".join(lines)


def _compact_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_name": turn.get("agent_name", ""),
        "role": turn.get("role", ""),
        "visible_findings": list(turn.get("visible_findings", []))[:5],
        "negative_findings": list(turn.get("negative_findings", []))[:4],
        "candidate_causes": list(turn.get("candidate_causes", []))[:4],
        "evidence_strength": turn.get("evidence_strength", ""),
        "ranked_differentials": list(turn.get("ranked_differentials", []))[:4],
        "why_primary": list(turn.get("why_primary", []))[:4],
        "why_not_primary": list(turn.get("why_not_primary", []))[:4],
        "decisive_missing_evidence": list(turn.get("decisive_missing_evidence", []))[:4],
        "today_actions": list(turn.get("today_actions", []))[:4],
        "control_options": list(turn.get("control_options", []))[:4],
        "observe_48h": list(turn.get("observe_48h", []))[:4],
        "escalation_triggers": list(turn.get("escalation_triggers", []))[:4],
        "management_timeline": list(turn.get("management_timeline", []))[:4],
        "low_risk_actions": list(turn.get("low_risk_actions", []))[:4],
        "environment_adjustments": list(turn.get("environment_adjustments", []))[:4],
        "followup_nodes": list(turn.get("followup_nodes", []))[:4],
        "prohibited_actions": list(turn.get("prohibited_actions", []))[:4],
        "overtreatment_risks": list(turn.get("overtreatment_risks", []))[:4],
        "undertreatment_risks": list(turn.get("undertreatment_risks", []))[:4],
        "confidence_boundary": list(turn.get("confidence_boundary", []))[:4],
        "citations": list(turn.get("citations", []))[:5],
    }


def _compact_shared_state(shared_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "consensus": list(shared_state.get("consensus", []))[:5],
        "conflicts": list(shared_state.get("conflicts", []))[:5],
        "unique_points": list(shared_state.get("unique_points", []))[:5],
        "next_focus": list(shared_state.get("next_focus", []))[:5],
        "safety_flags": list(shared_state.get("safety_flags", []))[:5],
        "working_diagnoses": list(shared_state.get("working_diagnoses", []))[:5],
        "open_questions": list(shared_state.get("open_questions", []))[:5],
        "evidence_gaps": list(shared_state.get("evidence_gaps", []))[:5],
        "uncertainty_score": shared_state.get("uncertainty_score", 0.5),
        "diagnosis_board": shared_state.get("diagnosis_board", {}),
        "evidence_board": shared_state.get("evidence_board", {}),
        "action_board": shared_state.get("action_board", {}),
        "risk_board": shared_state.get("risk_board", {}),
        "action_focus": list(shared_state.get("action_focus", []))[:5],
        "verification_tasks": list(shared_state.get("verification_tasks", []))[:5],
        "uncertainty_triggers": list(shared_state.get("uncertainty_triggers", []))[:5],
    }


def _expert_round_shared_state(shared_state: dict[str, Any], round_idx: int) -> dict[str, Any]:
    """专家提示用共享状态：第 2 轮起不再塞入「共识」长列表，避免每轮炒冷饭；从不包含 evidence_sufficiency 复读。"""
    boards = {
        "diagnosis_board": shared_state.get("diagnosis_board", {}),
        "evidence_board": shared_state.get("evidence_board", {}),
        "action_board": shared_state.get("action_board", {}),
        "risk_board": shared_state.get("risk_board", {}),
        "diagnosis_evidence": list(shared_state.get("diagnosis_evidence", []))[:8]
        if isinstance(shared_state.get("diagnosis_evidence"), list)
        else [],
    }
    core = {
        "working_diagnoses": list(shared_state.get("working_diagnoses", []))[:3],
        "next_focus": list(shared_state.get("next_focus", []))[:6],
        "conflicts": list(shared_state.get("conflicts", []))[:6],
        "open_questions": list(shared_state.get("open_questions", []))[:5],
        "uncertainty_score": shared_state.get("uncertainty_score", 0.5),
        **boards,
    }
    if round_idx <= 1:
        core["consensus"] = list(shared_state.get("consensus", []))[:5]
        core["unique_points"] = list(shared_state.get("unique_points", []))[:5]
    else:
        core["增量提示"] = (
            "以下为协调器更新后的板与子焦点。请只输出相对上一轮的新增推理、对分歧的回应或需要修正的点；"
            "不要复述已对齐的共识段落。"
        )
    return core


def _expert_output_spec(agent_name: str) -> dict[str, Any]:
    if agent_name == "diagnosis_evidence_officer":
        return {
            "required_fields": [
                "agent_name",
                "role",
                "visible_findings",
                "negative_findings",
                "candidate_causes",
                "evidence_strength",
                "citations",
            ],
            "field_rules": {
                "visible_findings": "只写肉眼可见或从输入事实明确得到的阳性观察。",
                "negative_findings": "只写缺失的关键征象或明确未观察到的线索。",
                "candidate_causes": "1 到 5 个候选，每个元素包含 name、why_like、why_unlike。",
                "evidence_strength": "一句话概括当前证据强弱与局限。",
            },
        }
    if agent_name == "differential_officer":
        return {
            "required_fields": [
                "agent_name",
                "role",
                "ranked_differentials",
                "why_primary",
                "why_not_primary",
                "decisive_missing_evidence",
                "citations",
            ],
            "field_rules": {
                "ranked_differentials": "1 到 5 个候选，按当前优先级排序，每个元素包含 name、why_supported、why_not_primary。",
                "why_primary": "支撑当前首选排第一的关键理由。",
                "why_not_primary": "说明其他候选为什么目前没有排第一。",
                "decisive_missing_evidence": "只写真正会改变当前排序的缺口。",
            },
        }
    if agent_name == "tomato_qa_expert":
        return {
            "required_fields": [
                "agent_name",
                "role",
                "today_actions",
                "control_options",
                "observe_48h",
                "escalation_triggers",
                "citations",
            ],
            "field_rules": {
                "today_actions": "只写今天就能执行的低风险动作。",
                "control_options": "写病种相关防治方案；药剂细节仅当知识库/病例摘录中存在时可整理写入，否则写原则级建议并注明依据缺口。",
                "observe_48h": "写 24 到 48 小时内要重点观察的变化。",
                "escalation_triggers": "写需要升级处理、复检或送检的触发条件。",
            },
        }
    if agent_name == "cultivation_management_officer":
        return {
            "required_fields": [
                "agent_name",
                "role",
                "management_timeline",
                "low_risk_actions",
                "environment_adjustments",
                "followup_nodes",
                "citations",
            ],
            "field_rules": {
                "management_timeline": "按先后顺序写管理节奏。",
                "low_risk_actions": "只写低风险、可逆、当下能落地的动作。",
                "environment_adjustments": "只写环境管理调整，如通风、湿度、灌溉、隔离。",
                "followup_nodes": "写后续复查节点和观察指标。",
            },
        }
    return {
        "required_fields": [
            "agent_name",
            "role",
            "citations",
        ],
        "field_rules": {},
    }


def _expert_special_rules(agent_name: str, round_idx: int = 1) -> list[str]:
    common = [
        "必须只返回严格 JSON，不要输出解释、标题、前后缀或 Markdown 代码块。",
        "所有用户可见内容必须使用中文。",
        "不要暴露任何内部模型名、组件名、路径、调试信息或系统术语。",
        "不要编造未在输入里出现的实验、化验、环境信息或病史。",
        "你是多智能体的一员：输出须可被其他角色检验；不要用空话假装已核实输入未提供的事实。",
        "不要在输出中反复铺陈「仅一张图/证据不足」等元叙事；系统已记录该约束。",
    ]
    single_image_rule = [
        "单张图像下只做可见形态与候选假设，不扩写未看见部位（如叶背、整株、茎果）。",
    ]
    role_rules = {
        "diagnosis_evidence_officer": [
            *(single_image_rule if round_idx <= 1 else []),
            "病理因果链保持简练，每条假设对应可见征象即可，勿写长段推演。",
            "不要给出治疗建议、环境管理建议或最终诊断结论。",
            "若知识库/病例中有与征象直接相关的句子，用短句纳入 candidate 或 citations；无则明确写无摘录可引，勿凭记忆补病害细节。",
            "须指出证据分别支持或削弱哪些候选，依据须指向输入中的可见事实或摘录。",
        ],
        "differential_officer": [
            "价值在证伪与压力测试：像验证专家一样假设「首选若错了，错在哪」；避免 rubber-stamp 式附和。",
            "必须把排序理由和未排第一的理由分开写。",
            "不要输出具体处置动作或通用管理建议。",
            "发现矛盾时写出具体缺失或冲突的征象/摘录位置，不写「可能不对」式空判。",
        ],
        "tomato_qa_expert": [
            "价值在把**输入内**防治信息整理成可执行步骤，而非扮演全知植保数据库。",
            "若知识库/病例摘录中出现具体药剂与用法：可整理进 control_options / today_actions，并在 citations 中给出可追溯短句。",
            "若摘录中**没有**可复核的药剂配方：禁止捏造名称与倍数；应写清「输入未提供登记药剂细节」，"
            "只给类别级建议（如保护性/治疗性杀菌剂的选择原则）并强调对照产品标签与本地农技。",
            "禁止用「【强制】必须写出某某药剂」类话术倒逼自己幻觉；宁短而诚实，勿长而编造。",
            "优先今日可执行与观察升级条件；不做主诊断排序，不自称裁决者。",
        ],
        "cultivation_management_officer": [
            "价值在环境与管理：温湿度、通风、灌溉、隔离、复查节奏。",
            "优先低风险、诊断偏差也不致药害的措施。",
            "不要重复病斑形态描述，不要重做诊断排序，不要推荐专性药剂。",
        ],
    }
    return common + role_rules.get(agent_name, [])


def build_expert_messages(
    *,
    expert: dict[str, str],
    case_text: str,
    caption: CaptionSchema,
    kb_evidence: list[dict[str, Any]],
    round_idx: int,
    shared_state: dict[str, Any],
) -> list[dict[str, str]]:
    agent_name = str(expert.get("agent_name", "")).strip()
    output_spec = _expert_output_spec(agent_name)
    round_context = (
        "这是第一轮讨论，请基于输入信息进行独立分析。"
        if round_idx == 1
        else (
            f"这是第 {round_idx} 轮讨论。请依据共享状态中的各「板」、分歧点与 next_focus 做增量分析："
            "补充新推理、回应质疑或收敛争议；不要整段复述上一轮已对齐的共识话术。"
        )
    )
    system = (
        f"{MODEL_CAPABILITY_PREAMBLE}\n\n"
        "你是番茄叶片病害多智能体诊断系统中的一位专家；其他专家并行分析同一病例，协调器负责汇总。\n"
        "你的单次回复是自包含的：不要假设读者读过你上一轮心路历程；用 JSON 字段承载可传递事实。\n"
        f"你的角色：{expert.get('role', '')}\n\n"
        f"{round_context}\n"
        "从本角色视角给出不可替代的分析；禁止用「作为专家我知道」替代输入中的证据。\n"
        "你必须只返回严格 JSON，字段必须符合给定输出协议，且不要出现任何额外字段。\n"
        "证据有限时：宁可明确标注不确定与缺口，也不要用详细虚构补满字段。"
    )
    session_boundary = ""
    if round_idx <= 1:
        session_boundary = (
            "【会话边界】当前通常仅有一张可用图像，系统已记录该客观限制；"
            "请勿在 JSON 各字段里反复陈述「只有一张图/证据不足」，把字数用在其体专业内容上。"
        )
    kb_evidence_formatted = _format_kb_evidence_for_expert(kb_evidence, agent_name)
    user = {
        "任务": f"第 {round_idx} 轮专家分析",
        "角色信息": expert,
        "病例描述": case_text,
        "视觉摘要": _caption_payload(caption),
        "知识库证据": kb_evidence_formatted,
        "共享状态摘要": _expert_round_shared_state(shared_state, round_idx),
        "输出协议": output_spec,
        "硬性约束": _expert_special_rules(agent_name, round_idx),
        "补充要求": [
            req
            for req in [
                session_boundary,
                "引用知识库或病例摘录时，把可追溯短句放进 citations；若下方无可用摘录，须在相应字段诚实说明，勿编造。",
                "信息不足也不要返回空对象：至少完成本角色最核心字段，可用短句标明「输入未覆盖」。",
                "除 JSON 外不要输出任何说明文字。",
            ]
            if req
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_round_summary_messages(
    *,
    round_turns: list[dict[str, Any]],
    round_idx: int,
    shared_state: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "你是多智能体轮次协调器，负责汇总本轮各专家的讨论结果。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "协调器价值在**理解与压缩**：先消化专家 JSON 中的可检验断言，再写入板与焦点；"
        "勿把「听起来专业」的长句当共识——没有证据绑定的句子不要升格为 consensus。\n"
        "你必须只返回严格 JSON，字段必须符合 CoordinatorSummarySchema。\n"
        "\n"
        "请做「增量汇总」，避免炒冷饭。\n"
        "- consensus：只写**本轮新形成或新加强**的一致判断（0–3 条短句）；若与上一轮相比无新共识，可留空列表。\n"
        "- conflicts：具体写出仍存分歧的点（可含专家视角差异），避免空泛。\n"
        "- unique_points：只收录本轮**新出现**的独立见解，不要重复上一轮已记录过的句子。\n"
        "- next_focus：只列**仍未解决**且值得下一轮继续推进的 1–4 条，不要把已闭合的话题再抄一遍。\n"
        "- evidence_sufficiency：不要每轮重复「单张图证据不足」套话；若无新的证据边界变化可填空字符串。\n"
        "各 board 字段应收敛为板载结构化信息，不要整段复述专家原文。"
    )
    user = {
        "任务": f"汇总第 {round_idx} 轮专家输出",
        "上一轮共享状态": _compact_shared_state(shared_state),
        "本轮专家输出": [_compact_turn(turn) for turn in round_turns],
        "汇总规则": [
            "working_diagnoses 只保留当前最主要的诊断名称，去重后输出。",
            "diagnosis_board 只放诊断链相关内容，不要混入管理动作。",
            "evidence_board 只放会改变排序或处理强度的缺口。",
            "action_board 只放今天动作、防治方案、48 小时观察和升级条件。",
            "risk_board 只放禁止动作、风险标记和置信边界。",
            "不要把「无法判断」之类空泛短句直接扩散到多个字段，除非绑定具体对象。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_final_messages(
    *,
    case_text: str,
    caption: CaptionSchema,
    decision_packet: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "你是最终诊断整合器。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "你必须根据给定决策包生成严格 JSON，字段必须符合 FinalDiagnosisSchema。\n"
        "整合时区分：视觉模型倾向、多专家共识、仍存分歧；不要把单一来源写成「已证实」。\n"
        "输出要明确、审慎、可执行，所有用户可见内容必须使用中文。"
    )
    user = {
        "病例描述": case_text,
        "视觉摘要": _caption_payload(caption),
        "最终决策包": decision_packet,
        "生成要求": [
            "top_diagnosis.name 必须与决策包中 diagnosis_summary.primary_candidate 一致：该字段以**视觉解读模块**（分类与融合排序）为优先来源，多智能体结论用于补充与鉴别；不要自造与之冲突的首位病名。",
            "top_diagnosis：name 为首位疑似病名；confidence 为**短中文**，必须区分两类含义："
            "（1）图像/模型分类相似度或倾向；（2）田间病原确诊把握（可与（1）相反：分类倾向高仍可确诊把握低）。"
            "不得用分类百分比直接等同于确诊置信度。",
            "candidates 只保留与当前判断最相关的候选。",
            "symptom_summary 只写观察归纳，不要重复完整报告。",
            "actions、monitoring_plan、prohibited_actions 要能直接执行。",
            "report_outline 必须覆盖以下六个固定章节：" + "、".join(REQUIRED_REPORT_SECTIONS),
            "confidence_statement 和 evidence_sufficiency 要清楚表达证据边界。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_safety_messages(*, final_result: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是安全审查员。\n"
        "你只负责审查最终建议中的安全边界，不负责重做诊断。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "对「具体药剂、剂量、浓度」类建议：若决策包未给出可复核来源，应倾向标记为需本地核实或改为更保守表述。\n"
        "你必须只返回严格 JSON，字段必须符合 SafetyReviewSchema。"
    )
    user = {
        "待审查结果": final_result,
        "审查重点": [
            "是否存在过于激进、不可逆或明显越权的动作。",
            "是否遗漏了必要的复查节点、禁做事项或风险提示。",
            "如需修正 revised_actions，应优先给出更保守、更低风险的替代方案。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_multi_agent_report_messages(
    *,
    case_text: str,
    caption: CaptionSchema,
    report_packet: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "你是农业病害报告写手。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "你只能使用给定材料中的事实与摘录；材料未写的防治细节不要凭训练记忆补全为「具体配方」。\n"
        "你需要根据给定材料生成完整 Markdown 报告；覆盖固定六节，语言具体、克制、中文自然。"
    )
    user = {
        "病例描述": case_text,
        "视觉摘要": _caption_payload(caption),
        "报告材料": report_packet,
        "章节要求": REQUIRED_REPORT_SECTIONS,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_multi_agent_report_section_messages(
    *,
    case_text: str,
    caption: CaptionSchema,
    section_packet: dict[str, Any],
    section_title: str,
    section_instruction: str,
    completed_sections: list[dict[str, str]],
) -> list[dict[str, str]]:
    system = (
        "你是农业病害报告的章节撰稿人。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "你当前只负责一个章节，只返回 JSON，字段需符合 MarkdownSectionSchema。\n"
        "section_markdown 必须为本节**完整正文**：至少两个自然段，由你自行安排叙述顺序、过渡与重点，"
        "把章节材料里的信息**内化**为连贯口语化书面语，不要逐字段复述或写成填空式清单。\n"
        "除「防治建议」外尽量少用编号列表；如需列表，嵌入段落语境中，不要整节只有条目。"
    )
    user = {
        "病例描述": case_text,
        "视觉摘要": _caption_payload(caption),
        "章节标题": section_title,
        "章节指令": section_instruction,
        "章节材料": section_packet,
        "已完成章节": completed_sections,
        "固定要求": [
            "只写当前章节，不越界扩写其他章节职责。",
            "参考 section_facts 与 shared_context，但用你自己的句子组织，避免照搬材料键名或 JSON 腔。",
            "段与段之间要有因果或递进，读起来像一篇报告的一节，而不是要点堆砌。",
            "诊断相关使用“首位疑似/候选方向”等审慎表述，不写确诊句式。",
            "防治建议先写可立即执行的低风险田间措施，证据不足时不写具体药剂配方。",
            "所有用户可见内容必须使用中文。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _section_formatting_hint(section_title: str) -> str:
    hints = {
        "基本信息": (
            "仅输出一个规范 Markdown 表格：表头行、对齐行（|:---|）、数据行各占独立一行，禁止把整张表粘成一行。"
            "行至少包含：作物；诊断病害（首位疑似）；病原或类别（不确定写「待复核」）；"
            "**图像分类相似度/模型倾向**（可写百分比，注明仅表示像哪一类）；"
            "**田间确诊把握**（低/中/高 或 审慎/疑似，与上一行区分）；病害阶段；受害部位；生育期与地点（未知则写未知）。"
            "表格之后至多 1～2 句收束，不得在本节展开长篇鉴别论述（长论述应放在「鉴别诊断」）。"
        ),
        "症状观察": (
            "用短条目列出肉眼可见的症状事实，每条一个观察点。"
            "只写观察到的现象，不写诊断结论。"
            "描述颜色、形态、分布、质感等客观特征。"
        ),
        "鉴别诊断": (
            "先给出规范 Markdown 表格（表头/对齐行/数据行分行书写），列：病害、可能性、依据。"
            "表格前可加不超过 3 句的导语，但不要重复「## 鉴别诊断」标题。"
            "依据列写短句即可，禁止在单元格内堆 HTML 或超长不换行段落。"
            "表格后可加极短的「若…则…」升级/降级提示，避免与表格内容整段重复。"
        ),
        "防治建议": (
            "用 ### 分子节（立即核查 / 药剂防治 / 农业防治），**每个三级标题后必须换行再接正文**。"
            "药剂防治用表格；农业防治用短条目。"
            "证据不足时写类别级建议并注明需本地复核。"
        ),
        "预后评估": (
            "输出 Markdown 表格，列为指标和评估。"
            "行包含：可治愈性、已受损部位恢复情况、周边风险、产量影响等。"
            "本节以表格为主，不必再写长段叙事。"
        ),
        "备注": (
            "用一段或引用块（>）收束全文：强调补证优先级、并行处理思路、何时线下复核。"
            "不复述前文表格内容；可点明证据边界与关键不确定点。"
        ),
    }
    return hints.get(section_title.strip(), "客观、简洁、中文书面语。")


def build_narrative_report_section_messages(
    *,
    case_text: str,
    caption: CaptionSchema,
    section_title: str,
    section_instruction: str,
    global_briefing: str,
    section_focus_markdown: str,
    prior_sections_excerpt: str,
) -> list[dict[str, str]]:
    """章节撰稿：自然语言材料 + 纯文本输出（由调用方约定不得输出 JSON）。"""
    system = (
        "你是农业植保技术简报撰稿人，读者含农户与一线植保；只撰写一个章节的中文正文。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "结构原则：六章顺排，各节各司其职，**禁止**重复总述。"
        "分工：基本信息/鉴别/预后以表格为主；症状观察用短条目；防治建议分子节+表格；备注为收束性短段或引用块。\n"
        "文风必须**客观、中性、书面**：陈述可观察事实与可执行条件，**避免**宣传腔、鸡汤句、通篇「你」的说教口吻。"
        "禁用俗套表达（含但不限于）：「记住」「牢牢」「主动权」「心里有数」「盯哨」「做扎实了心里就有底」「别走弯路」等。\n"
        "版式：以表格/条目为主的章节不必硬凑长段落；若写叙述段，**每段不超过约 6–8 句或 160 字**，段之间空一行。\n"
        "分数口径：分类分数高仅表示**照片更像哪一类**；不等于显微镜检或病原分离层面的「确诊概率」。\n"
        "Markdown 表格：`|表头|` 与 `|:---|` 与数据行之间必须有换行，禁止 `|单元格|| :--- |` 这种粘连写法。\n"
        "禁止输出 JSON、禁止 ##、禁止代码围栏；勿重复章节标题（系统已加 ##）。禁止成品中出现【】标签式小标题。\n"
        "禁用后台腔：支持链、鉴别要点、模型倾向、排序、会诊摘要等；改为「依据在照片上的哪些特征」「与何病区分」「未看清何细节则结论可能变」。\n"
        "可用加粗短语标题、`- ` 短列表；救治与观察用动词开头短句，并多写「若…则…」。\n"
        "诊断用「首位疑似/更像」等审慎表述，避免「确诊」、避免把分类分数写成病原确诊概率。\n"
        "多角色讨论材料请**内化改写**为中性田间表述，勿照抄内部结构或小标题。"
    )
    cap_vis = caption_to_knowledge_narrative(caption)
    if len(cap_vis) > 5000:
        cap_vis = cap_vis[:4980] + "…"
    fmt_hint = _section_formatting_hint(section_title)
    user_body = "\n".join(
        [
            f"章节标题：{section_title}",
            f"写作指令：{section_instruction}",
            "",
            "版式与分工（供你遵守，勿原样输出本段标题）：",
            fmt_hint,
            "",
            "硬性要求：材料中的阈值/条件可写，勿编造数字；前文已写过的句子勿整段复述；全节须明显分段（空行），避免墙式长段。",
            "",
            "病例描述：",
            (case_text or "（无）").strip(),
            "",
            "图像与模型侧（叙述体）：",
            cap_vis,
            "",
            "全局参考（含知识库与讨论摘录，请改写为客观、可执行的田间表述，勿抄内部用词）：",
            global_briefing.strip(),
            "",
            "本章专用知识段落（内化后写入本节，勿照抄）：",
            section_focus_markdown.strip() or "（无）",
            "",
            "已完成的前文节选（承上启下，避免重复）：",
            prior_sections_excerpt.strip() or "（本节为首节）",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_body},
    ]


def build_baseline_report_messages(
    *,
    case_text: str,
    caption: CaptionSchema,
    kb_evidence: list[dict[str, Any]],
    image_bytes: bytes | None = None,
) -> list[dict[str, str]]:
    system = (
        "你是单模型基线诊断与报告生成器。\n"
        f"{MODEL_CAPABILITY_PREAMBLE}\n"
        "你必须只返回严格 JSON，字段必须符合 BaselineOutputSchema。\n"
        "请根据病例描述、视觉摘要、知识库证据生成完整、可读的中文基线结果；无摘录时不要编造植保配方细节。"
    )
    user_payload: dict[str, Any] = {
        "病例描述": case_text,
        "视觉摘要": _caption_payload(caption),
        "知识库证据": kb_evidence,
        "输出要求": [
            "top_diagnosis 必须明确，不能写 unknown。",
            "report_outline 必须覆盖以下六个固定章节：" + "、".join(REQUIRED_REPORT_SECTIONS),
            "markdown_report 必须是完整中文 Markdown 报告。",
            "actions、risks、key_evidence 必须尽量具体。",
        ],
    }
    if image_bytes:
        user_payload["附加图像"] = {
            "encoding": "base64",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
