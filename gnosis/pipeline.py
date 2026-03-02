from openai import OpenAI
from .llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE
from .config import ALLOWED_CHARACTER_TAGS, DEFAULT_LLM_MODEL
from .models import CharacterExtraction, ScriptResult
from .chunking import ChunkingConfig, build_rolling_context, split_text_into_chunks
from .utils import remove_code_fences_regex

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import httpx
import json
import hashlib
from typing import Dict, List, Optional, Tuple

API_KEY = os.environ.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=600.0,
    http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)),
)

MAX_MERGED_SCRIPT_TEXT_CHARS = 80


def _parse_json_object(raw_content: str) -> dict:
    cleaned = remove_code_fences_regex(raw_content)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end >= start:
        cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM 返回的 JSON 根节点不是 object")
    return payload


def _build_request_key(
    model: str, messages: List[Dict[str, str]], response_format: Dict[str, str]
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_cache_path(
    cache_dir: str, stage: str, chunk_index: int, request_key: str
) -> str:
    stage_dir = os.path.join(cache_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    return os.path.join(stage_dir, f"{chunk_index:04d}_{request_key[:16]}.json")


def _load_cached_raw_response(cache_path: str) -> str:
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("raw_response")
    if isinstance(raw, str):
        return raw
    raise ValueError(f"缓存文件缺少 raw_response: {cache_path}")


def _save_raw_response(
    cache_path: str,
    stage: str,
    chunk_index: int,
    total_chunks: int,
    model: str,
    request_key: str,
    messages: List[Dict[str, str]],
    raw_response: str,
):
    payload = {
        "stage": stage,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "model": model,
        "request_key": request_key,
        "system_prompt": messages[0]["content"] if messages else "",
        "user_chunk": messages[-1]["content"] if messages else "",
        "raw_response": raw_response,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_raw_response(
    *,
    stage: str,
    chunk_index: int,
    total_chunks: int,
    model: str,
    messages: List[Dict[str, str]],
    cache_dir: str,
) -> Tuple[str, str, bool]:
    response_format = {"type": "json_object"}
    request_key = _build_request_key(model, messages, response_format)
    cache_path = _build_cache_path(cache_dir, stage, chunk_index, request_key)

    if os.path.exists(cache_path):
        print(f"chunk {chunk_index} cache hit")
        return _load_cached_raw_response(cache_path), cache_path, True

    print(model)
    print(response_format)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    raw_content = response.choices[0].message.content or ""
    _save_raw_response(
        cache_path=cache_path,
        stage=stage,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        model=model,
        request_key=request_key,
        messages=messages,
        raw_response=raw_content,
    )
    return raw_content, cache_path, False


def _normalize_custom_prompt(custom_prompt: str) -> str:
    value = (custom_prompt or "").strip()
    if value:
        return value
    return "无（使用通用规则）"


def _merge_consecutive_script_lines(
    script_lines: List[dict], max_text_chars: int = MAX_MERGED_SCRIPT_TEXT_CHARS
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


def run_pass1(
    text_segment,
    char_manager,
    chunking_config: ChunkingConfig = None,
    cache_dir: str = "data/llm_cache",
    pass1_custom_prompt: str = "",
):
    # Pass 1: 选角（按 chunk 迭代）
    chunking_config = chunking_config or ChunkingConfig()
    chunks = split_text_into_chunks(text_segment, chunking_config)
    if not chunks:
        return

    for chunk in chunks:
        known_str = char_manager.get_known_names()
        print(known_str)
        messages = [
            {
                "role": "system",
                "content": PASS1_PROMPT_TEMPLATE.format(
                    known_characters_str=known_str,
                    allowed_character_tags="\n".join(ALLOWED_CHARACTER_TAGS),
                    project_pass1_prompt=_normalize_custom_prompt(pass1_custom_prompt),
                ),
            },
            {"role": "user", "content": chunk.text},
        ]
        raw_content, cache_path, from_cache = _get_raw_response(
            stage="pass1",
            chunk_index=chunk.index,
            total_chunks=len(chunks),
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            cache_dir=cache_dir,
        )
        source = "cache" if from_cache else "llm"
        print(f"   [pass1] chunk {chunk.index}/{len(chunks)} <- {source}: {cache_path}")
        try:
            extraction_payload = _parse_json_object(raw_content)
        except Exception as e:
            raise ValueError(
                f"Pass1 第 {chunk.index} 段解析失败，原始响应已保存: {cache_path}"
            ) from e
        extraction = CharacterExtraction.model_validate(extraction_payload)
        for char in extraction.new_characters:
            char_manager.add_character(char)

    # 先把 pass1 角色结果持久化，再读取后进行声线分配，最后再次持久化
    char_manager.save_db()
    char_manager.load_db()
    char_manager.assign_voices()
    char_manager.save_db()


def run_pass2(
    text_segment,
    char_manager,
    chunking_config: ChunkingConfig = None,
    cache_dir: str = "data/llm_cache",
    pass2_workers: int = 4,
    pass2_custom_prompt: str = "",
):
    # Pass 2: 剧本（按 chunk 迭代 + 滚动上下文）
    chunking_config = chunking_config or ChunkingConfig()
    chunks = split_text_into_chunks(text_segment, chunking_config)
    characters_payload = [
        c.model_dump() for c in char_manager.characters.values()
    ]
    if not chunks:
        return {"characters": characters_payload, "script": []}

    updated_known = char_manager.get_known_names_and_gender()
    previous_context_map = {}
    previous_chunk_context = "无（这是第一段）"
    for chunk in chunks:
        previous_context_map[chunk.index] = previous_chunk_context
        previous_chunk_context = (
            build_rolling_context(chunk, chunking_config) or "无（这是第一段）"
        )

    def _process_chunk(chunk):
        messages = [
            {
                "role": "system",
                "content": PASS2_PROMPT_TEMPLATE.format(
                    available_characters_str=updated_known,
                    previous_chunk_context_str=previous_context_map[chunk.index],
                    chunk_index=chunk.index,
                    total_chunks=len(chunks),
                    project_pass2_prompt=_normalize_custom_prompt(pass2_custom_prompt),
                ),
            },
            {"role": "user", "content": chunk.text},
        ]
        raw_content, cache_path, from_cache = _get_raw_response(
            stage="pass2",
            chunk_index=chunk.index,
            total_chunks=len(chunks),
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            cache_dir=cache_dir,
        )
        source = "cache" if from_cache else "llm"
        print(f"   [pass2] chunk {chunk.index}/{len(chunks)} <- {source}: {cache_path}")
        try:
            script_payload = _parse_json_object(raw_content)
        except Exception as e:
            raise ValueError(
                f"Pass2 第 {chunk.index} 段解析失败，原始响应已保存: {cache_path}"
            ) from e
        chunk_script = ScriptResult.model_validate(script_payload)
        return chunk.index, [line.model_dump() for line in chunk_script.script]

    max_workers = max(1, min(int(pass2_workers or 1), len(chunks)))
    chunk_results = {}
    if max_workers == 1:
        for chunk in chunks:
            chunk_index, chunk_lines = _process_chunk(chunk)
            chunk_results[chunk_index] = chunk_lines
    else:
        print(f"   [pass2] 并发执行: workers={max_workers}, chunks={len(chunks)}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                chunk_index, chunk_lines = future.result()
                chunk_results[chunk_index] = chunk_lines

    script_lines = []
    for chunk in chunks:
        script_lines.extend(chunk_results.get(chunk.index, []))
    script_lines = _merge_consecutive_script_lines(script_lines)

    return {"characters": characters_payload, "script": script_lines}
