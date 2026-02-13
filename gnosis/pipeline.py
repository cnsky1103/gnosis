from openai import OpenAI
from .llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE
from .models import CharacterExtraction, ScriptResult

import os
import httpx

API_KEY = os.environ.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=600.0,
    http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)),
)


def run_pass1(text_segment, char_manager):
    # Pass 1: 选角
    known_str = char_manager.get_known_names()
    resp1 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": PASS1_PROMPT_TEMPLATE.format(known_characters_str=known_str),
            },
            {"role": "user", "content": text_segment},
        ],
        response_format={"type": "json_object"},
    )
    extraction = CharacterExtraction.model_validate_json(
        resp1.choices[0].message.content
    )
    for char in extraction.new_characters:
        char_manager.add_character(char)
    char_manager.save_db()


def run_pass2(text_segment, char_manager):
    # Pass 2: 剧本
    updated_known = char_manager.get_known_names()
    resp2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": PASS2_PROMPT_TEMPLATE.format(
                    available_characters_str=updated_known
                ),
            },
            {"role": "user", "content": text_segment},
        ],
        response_format={"type": "json_object"},
    )
    return ScriptResult.model_validate_json(resp2.choices[0].message.content).script
