import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

DEFAULT_PROJECT_PROMPT_FILE = "prompt.json"

PASS1_PROMPT_KEYS = (
    "pass1_prompt",
    "pass1_custom_prompt",
    "pass1",
    "prompt1",
)
PASS2_PROMPT_KEYS = (
    "pass2_prompt",
    "pass2_custom_prompt",
    "pass2",
    "prompt2",
)


@dataclass
class ProjectPromptOverrides:
    pass1_prompt: str = ""
    pass2_prompt: str = ""
    source_path: Optional[str] = None


def _pick_prompt(payload: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return ""


def load_project_prompt_overrides(
    project_root: str, file_name: str = DEFAULT_PROJECT_PROMPT_FILE
) -> ProjectPromptOverrides:
    prompt_path = os.path.join(project_root, file_name)
    if not os.path.isfile(prompt_path):
        return ProjectPromptOverrides()

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"project prompt 文件 JSON 格式错误: {prompt_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"project prompt 文件必须是 JSON 对象: {prompt_path}")

    return ProjectPromptOverrides(
        pass1_prompt=_pick_prompt(payload, PASS1_PROMPT_KEYS),
        pass2_prompt=_pick_prompt(payload, PASS2_PROMPT_KEYS),
        source_path=prompt_path,
    )
