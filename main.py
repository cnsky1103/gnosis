from gnosis.llm_director import SYSTEM_PROMPT_TEMPLATE
from gnosis.state_manager import CharacterManager
from gnosis.models import ChapterAnalysis
from gnosis.utils import remove_code_fences_regex
from gnosis.tts_engine import run_synthesis
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
        # with open("./data/input.txt", "r", encoding="utf-8") as f:
        #    raw_text = f.read()

        ## 这里调用 Rust 函数！
        # clean_content = gnosis_rs.clean_text(raw_text)
        # script = process_segment(clean_content)
        # with open("./data/character_db.json", "r", encoding="utf-8") as f:
        #    characters = json.loads(f.read())
        # with open("./data/out.txt", "r", encoding="utf-8") as f:
        #    data = json.loads(f.read())
        #    run_synthesis(data["script"], characters)
        # 假设 TTS 生成的音频都放在这个文件夹里
        audio_dir = "output_audio"

        # 我们最终要输出的有声书文件
        final_output = os.path.abspath("final_audiobook.mp3")

        print("等待所有音频生成完成...")
        # 你的 run_synthesis(script) 逻辑 ...

        print("交由 Rust 处理后期混音...")
        # 调用 Rust 函数：传入目录，输出路径，以及统一停顿时间（比如 400 毫秒）
        success = gnosis_rs.merge_audio(os.path.abspath(audio_dir), final_output, 200)

        if success:
            print("🎉 你的第一部有声书已经制作完成！快去听听看吧！")

    except Exception as _e:
        # 容错处理：可以把 raw_json 打印出来看看哪里错了
        traceback.print_exc()
