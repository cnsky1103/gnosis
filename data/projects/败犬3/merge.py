import os
from typing import Dict, List, Optional, Tuple
import json

from pathlib import Path

def get_sorted_files(dir_path, extension=".json"):
    # 获取目录下所有指定后缀的文件
    path = Path(dir_path)
    # 使用 sorted 进行排序，path.glob 会生成一个生成器
    files = sorted(path.glob(f"*{extension}"))
    
    # files 现在是 Path 对象的列表
    return [str(f) for f in files]

# 使用示例
files = get_sorted_files("./llm_cache/pass2")


def _merge_consecutive_script_lines(
    script_lines: List[dict], max_text_chars: int
) -> List[dict]:
    if max_text_chars < 1:
        return list(script_lines)

    merged: List[dict] = []
    current: Optional[dict] = None

    for raw_line in script_lines:
        if not isinstance(raw_line, dict):
            if current is not None:
                merged.append(current)
                current = None
            continue

        line = dict(raw_line)
        speaker = str(line.get("speaker", "")).strip()
        text = str(line.get("text", ""))

        if current is None:
            current = line
            continue

        current_speaker = str(current.get("speaker", "")).strip()
        current_text = str(current.get("text", ""))

        can_merge = (
            bool(speaker)
            and speaker == current_speaker
            and len(current_text + text) <= max_text_chars
        )
        if can_merge:
            current["text"] = current_text + text
            continue

        merged.append(current)
        current = line

    if current is not None:
        merged.append(current)
    return merged

script_lines = []
for file in files:
    with open(file, "r") as f:
        content = json.loads(json.load(f)["raw_response"])
        script_lines.extend(content["script"])

script_lines = _merge_consecutive_script_lines(script_lines, max_text_chars=80)
with open("./script.json", "w", encoding="utf-8") as f:
    json.dump({"script": script_lines}, f, ensure_ascii=False, indent=2)