import argparse
import asyncio
import json
import os
import sys
from gnosis.state_manager import CharacterManager
from gnosis.pipeline import run_pass1, run_pass2
from gnosis.tts_engine_sovits import tts_generate
import gnosis_rs

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

async def main():
    parser = argparse.ArgumentParser(description="Gnosis 有声书生产系统")
    parser.add_argument("step", choices=["extract", "script", "tts", "merge", "full"], 
                        help="运行步骤: extract(选角), script(剧本), tts(语音), merge(混音), full(全流程)")
    parser.add_argument("--input", default="novel.txt", help="输入的小说文本文件")
    parser.add_argument("--pause", type=int, default=400, help="句子间的停顿毫秒数")
    
    args = parser.parse_args()
    
    # 初始化管理器
    char_manager = CharacterManager(db_path="data/character_db.json", seeds_dir="./seeds")
    script_path = "data/script.json"
    audio_dir = "output_audio"

    # --- Step 1: 提取角色 ---
    if args.step in ["extract", "full"]:
        print("🔍 [Step 1] 正在分析角色并绑定种子...")
        text = load_text(args.input)
        run_pass1(text, char_manager) # 内部会自动 save_db
        print(f"✅ 角色库已更新: {len(char_manager.characters)} 个角色")

    # --- Step 2: 生成剧本 ---
    if args.step in ["script", "full"]:
        print("📝 [Step 2] 正在生成结构化剧本...")
        text = load_text(args.input)
        script_data = run_pass2(text, char_manager)
        with open(script_path, 'w', encoding='utf-8') as f:
            json.dump([line.model_dump() for line in script_data], f, ensure_ascii=False, indent=2)
        print(f"✅ 剧本已保存至: {script_path}")

    # --- Step 3: 语音生成 (TTS) ---
    if args.step in ["tts", "full"]:
        print("🎙️ [Step 3] 正在调用 GPT-SoVITS 生成音频...")
        if not os.path.exists(script_path):
            print("❌ 错误: 找不到剧本文件，请先运行 script 步骤")
            return
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_list = json.load(f)
        
        os.makedirs(audio_dir, exist_ok=True)
        for i, line in enumerate(script_list):
            file_path = os.path.join(audio_dir, f"{i:04d}.wav")
            if os.path.exists(file_path): continue # 跳过已存在的，方便断点续传
            
            print(f"   进度: {i+1}/{len(script_list)} -> {line['speaker']}")
            await tts_generate(line['text'], line['speaker'], char_manager, file_path)
        print("✅ 音频片段生成完毕")

    # --- Step 4: 合并混音 ---
    if args.step in ["merge", "full"]:
        print("🎚️ [Step 4] Rust 引擎正在混音并执行响度归一化...")
        # 1. 准备 Rust 需要的列表文件
        paths = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        list_file = os.path.join(audio_dir, "concat_list.txt")
        
        # 简单处理：这里也可以生成静音帧逻辑，或者直接交给 Rust
        with open(list_file, 'w') as f:
            for p in paths:
                f.write(f"file '{p}'\n")
        
        final_file = "final_audiobook.mp3"
        success = gnosis_rs.merge_audio_pro(os.path.abspath(audio_dir), final_file)
        
        if success:
            print(f"🎉 大功告成！最终成品: {final_file}")

if __name__ == "__main__":
    asyncio.run(main())
