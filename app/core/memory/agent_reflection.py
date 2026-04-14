"""Agent 反思模块 — 表观遗传记忆 (Epigenetic Memory)

植物的表观遗传修饰（如 DNA 甲基化、组蛋白修饰）能在不改变基因序列的前提下，
调整基因表达模式，使植物"记住"之前的胁迫经历并做出更适应性的响应。

本模块为每个 Agent 维护一份"表观修饰记录"：
- 记录 Agent 在历次诊断中的行为特征（如偏向性、准确度、独特贡献）
- 在后续诊断中注入这些经验，调整 Agent 的行为倾向
- 不改变 Agent 的核心人格（基因），但微调其表达方式
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class AgentReflection:
    """管理单个 Agent 的反思记忆。

    每次诊断结束后，协调器会调用 record_reflection 记录该 Agent 的表现。
    下次启动时，通过 build_experience_context 生成经验注入片段。
    """

    def __init__(self, memory_dir: str | Path, max_reflections: int = 50):
        self.memory_dir = Path(memory_dir) / "agent_reflections"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_reflections = max_reflections

    def _agent_file(self, agent_name: str) -> Path:
        return self.memory_dir / f"{agent_name}.jsonl"

    def record_reflection(
        self,
        agent_name: str,
        run_id: str,
        *,
        diagnosis_outcome: str,
        agent_contribution: str,
        behavioral_notes: list[str],
        accuracy_signal: str = "unknown",
        peer_feedback: list[str] | None = None,
        self_assessment: dict[str, Any] | None = None,
    ) -> None:
        """记录一次诊断中 Agent 的反思。

        Args:
            agent_name: Agent 标识
            run_id: 本次诊断运行 ID
            diagnosis_outcome: 最终诊断结论（用于事后对比）
            agent_contribution: Agent 本次的核心贡献概述
            behavioral_notes: 行为特征记录（如偏向性、遗漏、独特发现等）
            accuracy_signal: 准确度信号 (correct/partial/incorrect/unknown)
            peer_feedback: 来自其他 Agent 或协调器的反馈
            self_assessment: Agent 自我评估（可由 LLM 生成）
        """
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "diagnosis_outcome": diagnosis_outcome,
            "agent_contribution": agent_contribution,
            "behavioral_notes": behavioral_notes,
            "accuracy_signal": accuracy_signal,
            "peer_feedback": peer_feedback or [],
            "self_assessment": self_assessment or {},
        }
        filepath = self._agent_file(agent_name)
        with filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trim_if_needed(filepath)

    def load_reflections(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        filepath = self._agent_file(agent_name)
        if not filepath.exists():
            return []
        records: list[dict[str, Any]] = []
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
        return records[-limit:]

    def build_experience_context(self, agent_name: str, limit: int = 5) -> str:
        """为 Agent 构建经验注入上下文，添加到其 system prompt 中。"""
        reflections = self.load_reflections(agent_name, limit=limit)
        if not reflections:
            return ""

        lines = [
            "\n\n## 历史经验记忆（SAR — 系统获得性抗性）\n",
            "以下是你在过去诊断中积累的经验。这些经验帮助你避免重复犯错，并放大你的独特优势：\n",
        ]

        behavioral_patterns: list[str] = []
        accuracy_stats = {"correct": 0, "partial": 0, "incorrect": 0, "unknown": 0}

        for ref in reflections:
            signal = str(ref.get("accuracy_signal", "unknown")).strip()
            if signal in accuracy_stats:
                accuracy_stats[signal] += 1
            for note in ref.get("behavioral_notes", []):
                if note and note not in behavioral_patterns:
                    behavioral_patterns.append(note)

        total = sum(accuracy_stats.values())
        if total > 0:
            correct_rate = accuracy_stats["correct"] / total
            if correct_rate >= 0.7:
                lines.append(f"- 你在近 {total} 次诊断中准确率较高，保持当前分析策略\n")
            elif accuracy_stats["incorrect"] > accuracy_stats["correct"]:
                lines.append(
                    f"- 你在近 {total} 次诊断中有较多偏差，"
                    "请特别注意：不要过度自信，多标注不确定性\n"
                )

        if behavioral_patterns:
            lines.append("\n**行为倾向与修正建议：**\n")
            for note in behavioral_patterns[-6:]:
                lines.append(f"- {note}\n")

        recent = reflections[-1] if reflections else None
        if recent and recent.get("peer_feedback"):
            lines.append("\n**最近一次同行反馈：**\n")
            for fb in recent["peer_feedback"][:3]:
                lines.append(f"- {fb}\n")

        return "".join(lines)

    def build_reflection_prompt(
        self,
        agent_name: str,
        agent_output: dict[str, Any],
        peer_outputs: list[dict[str, Any]],
        final_diagnosis: str,
    ) -> str:
        """构建反思 prompt，让 LLM 为 Agent 生成自我反思。"""
        return (
            f"你是 {agent_name}，刚刚参与了一次多智能体植物病害诊断。\n\n"
            f"最终诊断结论为：{final_diagnosis}\n\n"
            "你的输出摘要：\n"
            f"{json.dumps(agent_output, ensure_ascii=False)[:800]}\n\n"
            "其他专家的输出摘要：\n"
            f"{json.dumps(peer_outputs, ensure_ascii=False)[:1200]}\n\n"
            "请从以下维度反思你的表现，返回严格 JSON：\n"
            "{\n"
            '  "contribution_summary": "你本次的核心贡献是什么",\n'
            '  "behavioral_notes": ["你注意到自己的哪些行为倾向（如偏向某种诊断、遗漏某类证据等）"],\n'
            '  "improvement_points": ["下次遇到类似病例时，你应该改进什么"],\n'
            '  "unique_value": "你提供了哪些其他专家没有提到的独特视角",\n'
            '  "peer_observations": ["你对其他专家输出的观察——他们有什么值得学习或需要改进的"]\n'
            "}"
        )

    def _trim_if_needed(self, filepath: Path) -> None:
        """保持文件不超过 max_reflections 条记录。"""
        records: list[str] = []
        with filepath.open("r", encoding="utf-8") as f:
            records = [line for line in f if line.strip()]
        if len(records) > self.max_reflections:
            with filepath.open("w", encoding="utf-8") as f:
                for line in records[-self.max_reflections:]:
                    f.write(line if line.endswith("\n") else line + "\n")
