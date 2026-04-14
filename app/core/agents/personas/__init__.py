"""Agent Personas — 植物免疫启发的多智能体人格定义

每个 agent 对应植物免疫系统中的一个功能角色：

- diagnosis_evidence_officer  → PRR（模式识别受体）：从原始视觉信号中提取可观察证据
- differential_officer        → ETI（效应子触发免疫）：对假设进行压力测试与鉴别
- tomato_qa_expert            → 效应蛋白（Effector）：将诊断转化为具体防治动作
- cultivation_management_officer → SAR 调控（系统获得性抗性调控）：环境管理与风险缓冲
"""

from app.core.agents.personas.diagnosis_evidence_officer import PERSONA as DIAGNOSIS_EVIDENCE_OFFICER
from app.core.agents.personas.differential_officer import PERSONA as DIFFERENTIAL_OFFICER
from app.core.agents.personas.tomato_qa_expert import PERSONA as TOMATO_QA_EXPERT
from app.core.agents.personas.cultivation_management_officer import PERSONA as CULTIVATION_MANAGEMENT_OFFICER

ALL_PERSONAS: dict[str, dict[str, str]] = {
    "diagnosis_evidence_officer": DIAGNOSIS_EVIDENCE_OFFICER,
    "differential_officer": DIFFERENTIAL_OFFICER,
    "tomato_qa_expert": TOMATO_QA_EXPERT,
    "cultivation_management_officer": CULTIVATION_MANAGEMENT_OFFICER,
}


def get_expert_definitions() -> list[dict[str, str]]:
    """聚合各 persona 模块的 role，供 orchestrator 注入专家 system 提示。"""
    return [
        {"agent_name": name, "role": persona["role"]}
        for name, persona in ALL_PERSONAS.items()
    ]
