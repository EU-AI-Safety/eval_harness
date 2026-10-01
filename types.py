from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class DatasetRow:
    row_id: str
    prompt: str
    target: str
    language: str
    eu_ai_act_category: str
    annex_section: str
    reasoning: Optional[str] = None
    reasoning2: Optional[str] = None
    is_machine_translation: bool = False
    source_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationRecord:
    row_id: str
    prompt: str
    target: str
    response: str
    language: str
    eu_ai_act_category: str
    annex_section: str
    is_machine_translation: bool
    model_name: str
    backend: str
    prompt_hash: str | None = None
    status: str = "success"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JudgeRecord:
    row_id: str
    target: str
    judge_label: str
    expected_behavior: str
    response: str
    prompt: str
    model_name: str
    judge_name: str
    language: str
    eu_ai_act_category: str
    annex_section: str
    is_machine_translation: bool
    raw_judge_output: str
    judge_explanation: str = ""
    parsing_status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)