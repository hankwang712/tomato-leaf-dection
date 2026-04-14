from __future__ import annotations

import re
from typing import Any

from app.core.agents.prompts import REQUIRED_REPORT_SECTIONS


REQUIRED_SECTIONS = list(REQUIRED_REPORT_SECTIONS)

MIN_MARKDOWN_CHARS = 200
INTERNAL_PROCESS_PATTERNS = (
    r"所有专家一致(?:认为|认定)",
    r"专家一致(?:认为|认定)",
    r"会诊摘要",
    r"内部(?:讨论|会诊)",
)


def _is_list_line(line: str) -> bool:
    return line.startswith(("- ", "* ", "+ ")) or bool(re.match(r"^\d+\.\s+", line))


def _is_heading_line(line: str) -> bool:
    return line.startswith("#")


def _is_table_line(line: str) -> bool:
    return "|" in line


def _find_section_position(text: str, section: str) -> int:
    position = text.find(section)
    return position


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"[。！？!?；;]\s*", _normalize_text(text))
    return [sentence for sentence in raw_sentences if len(sentence) >= 18]


def _repeated_sentences(current: str, previous_sections: list[dict[str, str]] | None = None) -> list[str]:
    if not previous_sections:
        return []
    previous_sentences: set[str] = set()
    for item in previous_sections:
        previous_text = _normalize_text(item.get("markdown", ""))
        previous_sentences.update(_split_sentences(previous_text))
    return [sentence for sentence in _split_sentences(current) if sentence in previous_sentences]


def validate_report_section(
    section_title: str,
    markdown: str,
    *,
    report_packet: dict[str, Any] | None = None,
    previous_sections: list[dict[str, str]] | None = None,
) -> None:
    text = _normalize_text(markdown)
    if len(text) < 40:
        raise ValueError(f"章节[{section_title}]内容过短")

    repeated = _repeated_sentences(text, previous_sections)
    if len(repeated) >= 3:
        raise ValueError(f"章节[{section_title}]与前文重复过多: {repeated[0]}")


def validate_markdown_report(markdown: str) -> None:
    text = str(markdown or "").strip()
    if len(text) < MIN_MARKDOWN_CHARS:
        raise ValueError("报告长度不足")

    previous_pos = -1
    for section in REQUIRED_SECTIONS:
        position = _find_section_position(text, section)
        if position < 0:
            raise ValueError(f"缺少必填章节：{section}")
        if position < previous_pos:
            raise ValueError("必填章节顺序错误")
        previous_pos = position

    for pattern in INTERNAL_PROCESS_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raise ValueError(f"报告暴露了内部讨论过程表述：{match.group(0)}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("报告正文为空")

    content_lines = [
        line for line in lines
        if not _is_heading_line(line) and not _is_table_line(line)
    ]
    list_lines = sum(1 for line in content_lines if _is_list_line(line))
    non_list = len(content_lines) - list_lines
    table_lines = sum(1 for line in lines if _is_table_line(line))

    if non_list < 2 and table_lines < 2:
        raise ValueError("报告缺少段落或表格内容")
