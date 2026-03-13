import argparse
import asyncio
import json
import os
from gnosis.srt import generate_srt_subtitles_precise
from gnosis.state_manager import CharacterManager
from gnosis.chunking import ChunkingConfig
from gnosis.project_prompt import ProjectPromptOverrides, load_project_prompt_overrides
from gnosis.tts.tts_utils import DEFAULT_SOVITS_URL
from gnosis.tts.tts_utils import delete_character_audio_segments
from gnosis.srt import build_precise_timeline, check_sample_rate, write_timeline_file
from gnosis.tts.tts_engine_factory import (create_cosyvoice_engine, create_sovits_engine)
from gnosis.utils import parse_script_payload
from gnosis.merge_audio import generate_precise_master_audio

DEFAULT_MIN_RATIO = 1200 / 1800
DEFAULT_MAX_RATIO = 2600 / 1800

DEFAULT_TTS_ENGINE = "cosyvoice"

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(path, text):
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def resolve_project_path(project_root, raw_path):
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(project_root, raw_path)


def resolve_input_path(project_root, input_arg):
    if os.path.isabs(input_arg):
        return input_arg
    project_candidate = os.path.join(project_root, input_arg)
    if os.path.exists(project_candidate):
        return project_candidate
    cwd_candidate = os.path.abspath(input_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return project_candidate


def build_chunking_config(chunk_size: int, rolling_paragraphs: int, rolling_max_chars: int):
    target_chars = max(1, chunk_size)
    min_chars = max(1, int(target_chars * DEFAULT_MIN_RATIO))
    max_chars = max(target_chars + 1, int(target_chars * DEFAULT_MAX_RATIO))
    return ChunkingConfig(
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        rolling_paragraphs=rolling_paragraphs,
        rolling_max_chars=rolling_max_chars,
    )


async def main():
    parser = argparse.ArgumentParser(description="Gnosis 有声书生产系统")
    parser.add_argument(
        "step",
        choices=["extract", "script", "tts", "merge", "full", "proofread"],
        help=(
            "运行步骤: extract(选角), script(剧本), tts(语音), merge(混音), "
            "proofread(剧本校对 Web), full(全流程)"
        ),
    )
    parser.add_argument(
        "--project",
        required=True,
        help="项目名（每本书一个 project，相关产物会隔离到 data/projects/<project>/）",
    )
    parser.add_argument(
        "--input",
        default="input.txt",
        help="输入小说文本文件路径。相对路径优先按项目目录解析，随后回退当前目录",
    )
    parser.add_argument("--pause", type=int, default=200, help="句子间的停顿毫秒数")
    parser.add_argument(
        "--pass2-chunk-size",
        type=int,
        default=1800,
        help="pass2 分段目标字数（按空行边界切分）",
    )
    parser.add_argument(
        "--pass1-chunk-size",
        type=int,
        default=None,
        help="pass1 分段目标字数；默认等于 pass2 的 5 倍",
    )
    parser.add_argument(
        "--pass2-workers",
        type=int,
        default=4,
        help="script 阶段并发数（run_pass2），默认 4",
    )
    parser.add_argument(
        "--rolling-paragraphs",
        type=int,
        default=2,
        help="传给下一段的上一段结尾段落数",
    )
    parser.add_argument(
        "--rolling-max-chars",
        type=int,
        default=500,
        help="滚动上下文最大字符数",
    )
    parser.add_argument(
        "--llm-cache-dir",
        default="llm_cache",
        help="LLM 原始响应缓存目录（相对路径按项目目录解析）",
    )
    parser.add_argument(
        "--sovits-url",
        default=DEFAULT_SOVITS_URL,
        help="GPT-SoVITS HTTP 地址",
    )
    parser.add_argument(
        "--tts-engine",
        default=DEFAULT_TTS_ENGINE,
        choices=["cosyvoice", "gpt-sovits"],
        help="TTS 引擎选择：默认 cosyvoice，可切换为 gpt-sovits",
    )
    parser.add_argument(
        "--tts-workers",
        type=int,
        default=8,
        help="TTS 并发线程数（仅 cosyvoice 生效，最小值 1）",
    )
    parser.add_argument(
        "--char",
        default="",
        help="TTS 仅生成指定角色名的台词（按 script.speaker 精确匹配）",
    )
    parser.add_argument(
        "--delete-char-audio",
        default="",
        help="TTS 结束后删除指定角色名的全部语音片段（按 script.speaker 精确匹配）",
    )
    parser.add_argument(
        "--cosyvoice-sample-rate",
        type=int,
        default=22050,
        help="CosyVoice 输出采样率（用于将 API 返回 PCM 包装为 wav）",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8765,
        help="proofread Web 服务监听端口",
    )

    args = parser.parse_args()
    if args.tts_workers < 1:
        parser.error("--tts-workers 必须 >= 1")
    if args.pass2_workers < 1:
        parser.error("--pass2-workers 必须 >= 1")
    if args.web_port < 1 or args.web_port > 65535:
        parser.error("--web-port 必须在 1..65535 之间")
    if args.char.strip() and args.step not in ["tts", "full"]:
        parser.error("--char 仅支持在 tts / full 步骤中使用")
    if args.delete_char_audio.strip() and args.step not in ["tts", "full"]:
        parser.error("--delete-char-audio 仅支持在 tts / full 步骤中使用")

    project_name = args.project

    project_root = os.path.abspath(os.path.join("data", "projects", project_name))
    os.makedirs(project_root, exist_ok=True)
    print(f"📁 当前 project: {project_name}")
    print(f"📂 project 目录: {project_root}")

    if args.step == "proofread":
        from gnosis.proofread_web import run_proofread_server

        run_proofread_server(
            project_name=project_name,
            port=args.web_port,
            projects_root=os.path.join("data", "projects"),
        )
        return

    script_path = os.path.join(project_root, "script.json")
    audio_dir = os.path.join(project_root, "output_audio")
    character_db_path = os.path.join(project_root, "character_db.json")
    llm_cache_dir = resolve_project_path(project_root, args.llm_cache_dir)
    project_input_path = os.path.join(project_root, "input.txt")
    final_file = os.path.join(project_root, "final_audiobook.wav")
    project_prompts = ProjectPromptOverrides()
    if args.step in ["extract", "script", "full"]:
        project_prompts = load_project_prompt_overrides(project_root)
        if project_prompts.source_path:
            print(f"🧩 已加载 project prompt: {project_prompts.source_path}")
        else:
            print("🧩 未找到 project prompt（prompt.json），使用默认主 prompt")

    pass2_chunking_config = build_chunking_config(
        chunk_size=args.pass2_chunk_size,
        rolling_paragraphs=args.rolling_paragraphs,
        rolling_max_chars=args.rolling_max_chars,
    )
    pass1_chunk_size = (
        args.pass1_chunk_size
        if args.pass1_chunk_size is not None
        else args.pass2_chunk_size * 5
    )
    pass1_chunking_config = build_chunking_config(
        chunk_size=pass1_chunk_size,
        rolling_paragraphs=args.rolling_paragraphs,
        rolling_max_chars=args.rolling_max_chars,
    )

    char_manager = None
    if args.step in ["extract", "script", "full"]:
        char_manager = CharacterManager(
            db_path=character_db_path
        )
    text = None
    if args.step in ["extract", "script", "full"]:
        input_path = resolve_input_path(project_root, args.input)
        text = load_text(input_path)
        save_text(project_input_path, text)
        print(f"📖 已使用原文: {os.path.abspath(input_path)}")
        print(f"🧾 已同步原文快照: {project_input_path}")

    # --- Step 1: 提取角色 ---
    if args.step in ["extract", "full"]:
        from gnosis.pipeline import run_pass1

        print("🔍 [Step 1] 正在分析角色并绑定种子...")
        run_pass1(
            text,
            char_manager,
            pass1_chunking_config,
            cache_dir=llm_cache_dir,
            pass1_custom_prompt=project_prompts.pass1_prompt,
        )  # 内部会自动 save_db
        print(f"✅ 角色库已更新: {len(char_manager.characters)} 个角色")

    # --- Step 2: 生成剧本 ---
    if args.step in ["script", "full"]:
        from gnosis.pipeline import run_pass2

        print("📝 [Step 2] 正在生成结构化剧本...")
        script_data = run_pass2(
            text,
            char_manager,
            pass2_chunking_config,
            cache_dir=llm_cache_dir,
            pass2_workers=args.pass2_workers,
            pass2_custom_prompt=project_prompts.pass2_prompt,
        )
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 剧本已保存至: {script_path}")

    # --- Step 3: 语音生成 (TTS) ---
    if args.step in ["tts", "full"]:
        print(f"🎙️ [Step 3] 正在调用 {args.tts_engine} 生成音频...")
        if not os.path.exists(script_path):
            print("❌ 错误: 找不到剧本文件，请先运行 script 步骤")
            return


        with open(script_path, "r", encoding="utf-8") as f:
            script_payload = json.load(f)
            characters, script_list = parse_script_payload(script_payload)
        delete_target = args.delete_char_audio.strip()
        if delete_target:
            print("🧹 [Step 3] 仅删除模式：跳过 TTS 生成")
            delete_stats = delete_character_audio_segments(
                audio_dir=audio_dir,
                script_lines=script_list,
                character_name=delete_target,
            )
            if delete_stats["matched_lines"] == 0:
                print(
                    "⚠️ 删除阶段未命中角色:"
                    f" {delete_target}（未删除任何文件）"
                )
            else:
                print(
                    "🧹 删除角色语音完成:"
                    f" {delete_stats['target']}，命中台词 {delete_stats['matched_lines']} 句，"
                    f" 删除文件 {delete_stats['deleted_files']} 个"
                )
            print("✅ 删除阶段执行完毕")
            return

        if args.tts_engine == "cosyvoice":
            tts_engine = create_cosyvoice_engine()
        elif args.tts_engine == "sovits":
            tts_engine = create_sovits_engine(args.sovits_url)

        os.makedirs(audio_dir, exist_ok=True)
        await tts_engine.generate_script_tts(script_path,audio_dir, char=args.char.strip())

        print("✅ 音频片段生成完毕")

    # --- Step 4: 合并混音 ---
    if args.step in ["merge", "full"]:

        print("🎚️ [Step 4] Rust 引擎正在混音并执行响度归一化...")
        if not os.path.exists(audio_dir):
            print("❌ 错误: 找不到音频目录，请先运行 tts 步骤")
            return

        precise_timeline = build_precise_timeline(os.path.abspath(audio_dir), args.pause)
        timeline_file = f"{os.path.splitext(final_file)[0]}.timeline.json"
        write_timeline_file(os.path.abspath(timeline_file), precise_timeline)
        print(
            "🧭 已生成样本级时间轴:"
            f" {timeline_file} (sr={precise_timeline['sample_rate']}Hz, "
            f"pause={precise_timeline['pause_samples']} samples)"
        )

        success = generate_precise_master_audio(timeline_payload=precise_timeline, output_master_path=final_file)

        if success:
            #check_sample_rate(final_file, precise_timeline)

            print(f"🎉 大功告成！最终成品: {final_file}")
            subtitle_file = (
                resolve_project_path(project_root, "final_subtitle.srt")
            )
            if os.path.exists(script_path):
                with open(script_path, "r", encoding="utf-8") as f:
                    script_payload = json.load(f)
                    _, script_list = parse_script_payload(script_payload)
                subtitle_stats = generate_srt_subtitles_precise(
                    script_lines=script_list,
                    subtitle_path=os.path.abspath(subtitle_file),
                    timeline_payload=precise_timeline,
                )
                print(
                    f"📝 已生成同步字幕: {subtitle_file} "
                    f"(共 {subtitle_stats['subtitle_count']} 条，"
                    f"样本级时间轴，sr={subtitle_stats['sample_rate']}Hz)"
                )
            else:
                print("⚠️ 未找到剧本文件，跳过字幕生成")


if __name__ == "__main__":
    asyncio.run(main())
