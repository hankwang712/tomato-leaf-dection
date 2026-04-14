"""情景记忆模块 — 免疫记忆 B 细胞 (Memory B-Cell)

在适应性免疫中，记忆 B 细胞保存着对特定抗原的"快照"。
当相同或相似抗原再次出现时，记忆 B 细胞能跳过初次免疫的缓慢探索过程，
直接产生高亲和力抗体，实现快速、精准的二次应答。

本模块为每次完整诊断保存一份"情景快照"：
- 包含视觉特征、诊断路径、最终结论、专家共识/分歧等完整上下文
- 支持基于症状相似度的快速检索
- 为后续相似病例提供"免疫记忆"——跳过探索直接建议参考路径
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class EpisodicCaseMemory:
    """完整病例的情景记忆管理。

    与 CaseLibrary 的区别：
    - CaseLibrary 存储的是结构化的病例记录，面向检索和评分
    - EpisodicCaseMemory 存储的是诊断过程的"完整快照"，面向经验复用
    包含：Agent 间的讨论路径、分歧解决过程、最终如何收敛到结论
    """

    def __init__(self, memory_dir: str | Path, max_episodes: int = 200):
        self.memory_dir = Path(memory_dir) / "episodic_memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.memory_dir / "episodes.jsonl"
        self.max_episodes = max_episodes

    def store_episode(
        self,
        run_id: str,
        *,
        visual_signature: dict[str, Any],
        mechanism_tags: dict[str, Any],
        diagnosis_path: list[dict[str, Any]],
        final_diagnosis: str,
        confidence_level: str,
        agent_consensus: dict[str, Any],
        key_evidence: list[str],
        key_conflicts: list[str],
        resolution_strategy: str,
        outcome_quality: str = "unknown",
    ) -> None:
        """存储一次诊断的完整情景。

        Args:
            visual_signature: 视觉特征摘要（颜色、形态、分布等枚举值）
            mechanism_tags: 机制标签
            diagnosis_path: 诊断路径摘要 [{"round": 1, "consensus": [...], "conflicts": [...]}]
            final_diagnosis: 最终诊断名称
            confidence_level: 置信水平描述
            agent_consensus: Agent 共识状态 {"agreed": [...], "disagreed": [...]}
            key_evidence: 关键证据列表
            key_conflicts: 关键分歧列表
            resolution_strategy: 分歧解决策略描述
            outcome_quality: 结果质量评估 (high/medium/low/unknown)
        """
        episode = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "visual_signature": visual_signature,
            "mechanism_tags": mechanism_tags,
            "diagnosis_path": diagnosis_path[:5],
            "final_diagnosis": final_diagnosis,
            "confidence_level": confidence_level,
            "agent_consensus": agent_consensus,
            "key_evidence": key_evidence[:10],
            "key_conflicts": key_conflicts[:5],
            "resolution_strategy": resolution_strategy,
            "outcome_quality": outcome_quality,
        }
        with self.episodes_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        self._trim_if_needed()

    def retrieve_similar(self, visual_signature: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
        """检索视觉特征相似的历史情景。"""
        episodes = self._load_all()
        if not episodes:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in episodes:
            score = self._visual_similarity(visual_signature, ep.get("visual_signature", {}))
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit] if _ > 0.15]

    def build_memory_context(self, visual_signature: dict[str, Any], limit: int = 2) -> str:
        """构建记忆上下文，注入到多智能体讨论中。"""
        similar = self.retrieve_similar(visual_signature, limit=limit)
        if not similar:
            return ""

        lines = [
            "\n\n## 免疫记忆（类似历史病例经验）\n",
            "系统曾处理过以下相似症状的病例。这些经验供参考，但当前病例可能有不同的病因：\n",
        ]
        for i, ep in enumerate(similar, 1):
            diag = ep.get("final_diagnosis", "未知")
            conf = ep.get("confidence_level", "")
            evidence = ep.get("key_evidence", [])[:3]
            conflicts = ep.get("key_conflicts", [])[:2]
            strategy = ep.get("resolution_strategy", "")

            lines.append(f"\n**历史病例 {i}** — 最终诊断：{diag}（{conf}）\n")
            if evidence:
                lines.append(f"  关键证据：{'; '.join(evidence)}\n")
            if conflicts:
                lines.append(f"  曾有分歧：{'; '.join(conflicts)}\n")
            if strategy:
                lines.append(f"  解决策略：{strategy}\n")

        lines.append(
            "\n**注意**：历史经验仅供参考路径，不可替代对当前病例的独立分析。"
            "如果当前症状与历史病例有差异，以当前观察为准。\n"
        )
        return "".join(lines)

    def _visual_similarity(self, sig_a: dict[str, Any], sig_b: dict[str, Any]) -> float:
        """基于视觉特征集合的 Jaccard 相似度。"""
        if not sig_a or not sig_b:
            return 0.0

        score = 0.0
        weight_total = 0.0
        for key, weight in [
            ("color", 0.15), ("tissue_state", 0.2), ("spot_shape", 0.15),
            ("boundary", 0.1), ("distribution_position", 0.1),
            ("distribution_pattern", 0.1), ("morph_change", 0.1), ("co_signs", 0.1),
        ]:
            set_a = set(str(v) for v in sig_a.get(key, []) if str(v).strip())
            set_b = set(str(v) for v in sig_b.get(key, []) if str(v).strip())
            if set_a and set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
                score += jaccard * weight
                weight_total += weight
            elif not set_a and not set_b:
                weight_total += weight

        return score / weight_total if weight_total > 0 else 0.0

    def _load_all(self) -> list[dict[str, Any]]:
        if not self.episodes_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.episodes_file.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
        return records

    def _trim_if_needed(self) -> None:
        records = self._load_all()
        if len(records) > self.max_episodes:
            with self.episodes_file.open("w", encoding="utf-8") as f:
                for record in records[-self.max_episodes:]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
