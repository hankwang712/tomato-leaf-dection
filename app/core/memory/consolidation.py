"""记忆整合模块 — 免疫亲和力成熟 (Affinity Maturation)

在适应性免疫中，B 细胞经过体细胞超突变和亲和力选择，
逐步优化其抗体对抗原的识别能力。每一轮选择都保留匹配更好的变体。

本模块模拟这一过程：
- 从多次诊断经历中提取跨病例的稳定模式
- 识别重复出现的 Agent 行为偏差并生成校正信号
- 将分散的情景记忆整合为结构化的诊断经验知识
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryConsolidation:
    """跨病例的记忆整合与模式提炼。

    定期（或达到阈值时）对情景记忆和 Agent 反思进行整合：
    1. 提取高频出现的症状-诊断关联模式
    2. 识别 Agent 反复出现的行为偏差
    3. 生成"诊断指南修正"——类似免疫系统的亲和力成熟
    """

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir) / "consolidated"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_file = self.memory_dir / "diagnostic_patterns.json"
        self.agent_biases_file = self.memory_dir / "agent_biases.json"

    def consolidate_from_episodes(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """从情景记忆中整合诊断模式。"""
        if not episodes:
            return {"patterns": [], "agent_biases": {}, "timestamp": datetime.now().isoformat()}

        diagnosis_counter: Counter[str] = Counter()
        stressor_counter: Counter[str] = Counter()
        symptom_diagnosis_links: list[dict[str, Any]] = []

        for ep in episodes:
            diag = str(ep.get("final_diagnosis", "")).strip()
            if diag:
                diagnosis_counter[diag] += 1

            tags = ep.get("mechanism_tags", {})
            for sc in tags.get("stressor_class", []):
                stressor_counter[str(sc).strip()] += 1

            vis_sig = ep.get("visual_signature", {})
            if diag and vis_sig:
                symptom_diagnosis_links.append({
                    "diagnosis": diag,
                    "visual_keys": self._extract_visual_keys(vis_sig),
                    "quality": ep.get("outcome_quality", "unknown"),
                })

        patterns = self._extract_stable_patterns(symptom_diagnosis_links)
        result = {
            "patterns": patterns,
            "diagnosis_frequency": dict(diagnosis_counter.most_common(20)),
            "stressor_frequency": dict(stressor_counter.most_common(10)),
            "total_episodes": len(episodes),
            "timestamp": datetime.now().isoformat(),
        }
        self._save_patterns(result)
        return result

    def consolidate_agent_biases(
        self,
        agent_reflections: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """从 Agent 反思记录中整合行为偏差。"""
        biases: dict[str, dict[str, Any]] = {}

        for agent_name, reflections in agent_reflections.items():
            if not reflections:
                continue

            all_notes: list[str] = []
            accuracy_counts = {"correct": 0, "partial": 0, "incorrect": 0, "unknown": 0}

            for ref in reflections:
                for note in ref.get("behavioral_notes", []):
                    if note:
                        all_notes.append(note)
                signal = str(ref.get("accuracy_signal", "unknown")).strip()
                if signal in accuracy_counts:
                    accuracy_counts[signal] += 1

            note_counter = Counter(all_notes)
            recurring_patterns = [
                note for note, count in note_counter.most_common(5) if count >= 2
            ]

            total = sum(accuracy_counts.values())
            biases[agent_name] = {
                "recurring_patterns": recurring_patterns,
                "accuracy_stats": accuracy_counts,
                "accuracy_rate": accuracy_counts["correct"] / total if total > 0 else 0.0,
                "sample_size": total,
                "correction_hints": self._generate_correction_hints(
                    recurring_patterns, accuracy_counts
                ),
            }

        self._save_agent_biases(biases)
        return biases

    def load_patterns(self) -> dict[str, Any]:
        if not self.patterns_file.exists():
            return {}
        try:
            with self.patterns_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def load_agent_biases(self) -> dict[str, dict[str, Any]]:
        if not self.agent_biases_file.exists():
            return {}
        try:
            with self.agent_biases_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def build_system_memory_context(self) -> str:
        """构建系统级记忆上下文，注入到协调器中。"""
        patterns = self.load_patterns()
        biases = self.load_agent_biases()

        if not patterns and not biases:
            return ""

        lines = ["\n\n## 系统诊断经验（亲和力成熟知识）\n"]

        if patterns.get("patterns"):
            lines.append("\n**稳定的症状-诊断关联模式：**\n")
            for p in patterns["patterns"][:5]:
                lines.append(
                    f"- {p.get('visual_pattern', '?')} → "
                    f"常见诊断：{p.get('common_diagnosis', '?')} "
                    f"（出现 {p.get('frequency', 0)} 次）\n"
                )

        if biases:
            lines.append("\n**Agent 行为校正信号：**\n")
            for agent_name, bias_info in biases.items():
                hints = bias_info.get("correction_hints", [])
                if hints:
                    lines.append(f"- {agent_name}：{'; '.join(hints[:2])}\n")

        return "".join(lines)

    @staticmethod
    def _extract_visual_keys(visual_signature: dict[str, Any]) -> str:
        keys: list[str] = []
        for field in ["color", "tissue_state", "spot_shape", "boundary"]:
            values = visual_signature.get(field, [])
            if values:
                keys.extend(str(v) for v in values if str(v).strip())
        return "+".join(sorted(keys)) if keys else "unknown"

    @staticmethod
    def _extract_stable_patterns(
        links: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """提取重复出现的视觉特征→诊断关联。"""
        pattern_counter: Counter[tuple[str, str]] = Counter()
        for link in links:
            quality = link.get("quality", "unknown")
            if quality == "low":
                continue
            vk = link.get("visual_keys", "")
            diag = link.get("diagnosis", "")
            if vk and diag:
                pattern_counter[(vk, diag)] += 1

        return [
            {
                "visual_pattern": vk,
                "common_diagnosis": diag,
                "frequency": count,
            }
            for (vk, diag), count in pattern_counter.most_common(10)
            if count >= 2
        ]

    @staticmethod
    def _generate_correction_hints(
        recurring_patterns: list[str],
        accuracy_counts: dict[str, int],
    ) -> list[str]:
        hints: list[str] = []
        total = sum(accuracy_counts.values())
        if total >= 5 and accuracy_counts.get("incorrect", 0) / total > 0.3:
            hints.append("准确率偏低，建议增加不确定性标注并减少过度自信的表述")
        if total >= 5 and accuracy_counts.get("correct", 0) / total > 0.8:
            hints.append("准确率较高，当前分析策略有效，可适当增加推断深度")
        for pattern in recurring_patterns[:3]:
            hints.append(f"反复出现的行为：{pattern}")
        return hints

    def _save_patterns(self, data: dict[str, Any]) -> None:
        with self.patterns_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_agent_biases(self, data: dict[str, dict[str, Any]]) -> None:
        with self.agent_biases_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
