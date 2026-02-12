from gnosis.llm_director import SYSTEM_PROMPT_TEMPLATE
from gnosis.state_manager import CharacterManager
from gnosis.models import ChapterAnalysis
from gnosis.utils import remove_code_fences_regex
from openai import OpenAI

import os
import json
import httpx
import traceback

import gnosis_rs

API_KEY = os.environ.get("API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=600.0,
    http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)),
)

char_manager = CharacterManager()


def process_segment(text_segment):
    # 1. 准备 Prompt 上下文
    known_chars_str = char_manager.get_known_names()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(known_characters_str=known_chars_str)

    # 2. 调用 LLM
    response = client.chat.completions.create(
        model="deepseek-v3.2",  # 推荐 v3.2 如果有
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析以下片段:\n---\n{text_segment}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    # 3. 解析与验证
    raw_json = response.choices[0].message.content
    raw_json = remove_code_fences_regex(raw_json)
    print(raw_json)
    analysis = ChapterAnalysis.model_validate_json(raw_json)

    # 4. 关键步骤：更新全局角色库
    db_updated = False
    for new_char in analysis.new_characters:
        if char_manager.add_character(new_char):
            db_updated = True

    if db_updated:
        char_manager.save_db()  # 只有当有新人时才写盘

    return analysis.script


if __name__ == "__main__":
    try:
        with open("./data/prelude.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 这里调用 Rust 函数！
        clean_content = gnosis_rs.clean_text(raw_text)
        script = process_segment(clean_content)

        # 打印结果，你会发现“老人”被自动注册了，而“温水和彦”复用了旧的设定
        print(
            json.dumps([s.model_dump() for s in script], ensure_ascii=False, indent=2)
        )
    except Exception as _e:
        # 容错处理：可以把 raw_json 打印出来看看哪里错了
        traceback.print_exc()
