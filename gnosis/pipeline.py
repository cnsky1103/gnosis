from openai import OpenAI
from .llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE
from .models import CharacterExtraction, ScriptResult
from .chunking import ChunkingConfig, build_rolling_context, split_text_into_chunks
from .utils import remove_code_fences_regex

import os
import httpx
import json
import hashlib
from typing import Dict, List, Tuple

API_KEY = os.environ.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=600.0,
    http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)),
)


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
    model: str, user_chunk: str, response_format: Dict[str, str]
) -> str:
    payload = {
        "model": model,
        "user_chunk": user_chunk,
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
    user_chunk = messages[-1]["content"] if messages else ""
    request_key = _build_request_key(model, user_chunk, response_format)
    cache_path = _build_cache_path(cache_dir, stage, chunk_index, request_key)

    if os.path.exists(cache_path):
        print(f"chunk {chunk_index} cache hit")
        return _load_cached_raw_response(cache_path), cache_path, True

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


def run_pass1(
    text_segment,
    char_manager,
    chunking_config: ChunkingConfig = None,
    cache_dir: str = "data/llm_cache",
):
    # Pass 1: 选角（按 chunk 迭代）
    chunking_config = chunking_config or ChunkingConfig()
    chunks = split_text_into_chunks(text_segment, chunking_config)
    if not chunks:
        return

    for chunk in chunks:
        known_str = char_manager.get_known_names()
        messages = [
            {
                "role": "system",
                "content": PASS1_PROMPT_TEMPLATE.format(known_characters_str=known_str),
            },
            {"role": "user", "content": chunk.text},
        ]
        raw_content, cache_path, from_cache = _get_raw_response(
            stage="pass1",
            chunk_index=chunk.index,
            total_chunks=len(chunks),
            model="deepseek-v3.2",
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
    previous_chunk_context = "无（这是第一段）"
    script_lines = []

    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": PASS2_PROMPT_TEMPLATE.format(
                    available_characters_str=updated_known,
                    previous_chunk_context_str=previous_chunk_context,
                    chunk_index=chunk.index,
                    total_chunks=len(chunks),
                ),
            },
            {"role": "user", "content": chunk.text},
        ]
        raw_content, cache_path, from_cache = _get_raw_response(
            stage="pass2",
            chunk_index=chunk.index,
            total_chunks=len(chunks),
            model="deepseek-v3.2",
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
        script_lines.extend([line.model_dump() for line in chunk_script.script])
        previous_chunk_context = (
            build_rolling_context(chunk, chunking_config) or "无（这是第一段）"
        )

    return {"characters": characters_payload, "script": script_lines}
