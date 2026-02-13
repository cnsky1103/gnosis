# novel_cast/tts_engine.py
import edge_tts
import os
import asyncio
from .config import ARCHETYPE_TO_EDGE_TTS


async def generate_audio_fragment(text, voice_id, output_path):
    """
    调用 Edge-TTS 生成单个音频文件
    """
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating {output_path}: {e}")
        return False


async def batch_synthesize(script_data, output_dir="output_audio"):
    """
    批量处理整个剧本列表
    script_data: 也就是 DeepSeek 返回的那个 List[Dict]
    """
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []

    for idx, line in enumerate(script_data):
        # 1. 确定文件名 (001_narrator.mp3, 002_wenshui.mp3)
        # 补零很重要，保证后续 Rust 合成时的顺序
        filename = f"{idx:04d}_{line['speaker']}.mp3"
        filepath = os.path.join(output_dir, filename)

        # 2. 选角
        voice_id = voice_map[line["speaker"]]

        # 3. 调整语速/语调 (可选优化)
        # 比如：如果是 sad，可以让语速慢一点 (Edge-TTS 支持 rate="-10%")
        # 这里先做最简单的

        print(f"正在生成 [{idx}] {line['speaker']}: {line['text'][:10]}...")

        # 4. 创建异步任务
        task = generate_audio_fragment(line["text"], voice_id, filepath)
        tasks.append(task)

    # 5. 并发执行所有生成任务 (速度极快)
    await asyncio.gather(*tasks)
    print(f"🎉 全部音频生成完毕！保存在 {output_dir}")


voice_map = {}


# 供外部调用的同步入口
def run_synthesis(script_data, character_list):
    for c in character_list:
        voice_map[c["name"]] = ARCHETYPE_TO_EDGE_TTS[c["voice_archetype"]]

    try:
        return asyncio.run(batch_synthesize(script_data))
    except RuntimeError as e:
        print(e)
