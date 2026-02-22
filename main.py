import argparse
import asyncio
import json
import os
import subprocess
import wave
from gnosis.state_manager import CharacterManager
from gnosis.chunking import ChunkingConfig
from gnosis.tts_engine import (
    DEFAULT_SOVITS_URL,
    DEFAULT_SWITCH_URL,
    build_voice_registry,
    switch_character,
    tts_generate,
)

DEFAULT_MIN_RATIO = 1200 / 1800
DEFAULT_MAX_RATIO = 2600 / 1800
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(path, text):
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def validate_project_name(project_name):
    value = (project_name or "").strip()
    if not value:
        raise ValueError("--project 不能为空")
    if os.path.isabs(value):
        raise ValueError("--project 不能是绝对路径")
    separators = [os.sep]
    if os.altsep:
        separators.append(os.altsep)
    if any(sep in value for sep in separators if sep):
        raise ValueError("--project 仅支持项目名，不可包含路径分隔符")
    if value in {".", ".."}:
        raise ValueError("--project 不能是 . 或 ..")
    return value


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


def parse_script_payload(script_payload):
    if isinstance(script_payload, dict) and isinstance(script_payload.get("script"), list):
        characters = script_payload.get("characters", [])
        if characters is None:
            characters = []
        if not isinstance(characters, list):
            raise ValueError("script.json 中 characters 必须是数组")
        return characters, script_payload["script"]
    if isinstance(script_payload, list):
        # 兼容旧格式：仅有 script 列表时，角色表置空并回退默认声线
        return [], script_payload
    raise ValueError("script.json 格式不正确，期望为 {'characters': [...], 'script': [...]} 或 [...]")


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


def _is_segment_file(path):
    if not os.path.isfile(path):
        return False
    file_name = os.path.basename(path)
    if file_name.startswith("silence_"):
        return False
    extension = os.path.splitext(file_name)[1].lower()
    return extension in SUPPORTED_AUDIO_EXTENSIONS


def _collect_sorted_segments(audio_dir):
    paths = [
        os.path.join(audio_dir, name)
        for name in os.listdir(audio_dir)
        if _is_segment_file(os.path.join(audio_dir, name))
    ]
    paths.sort()
    return paths


def _probe_duration_ms(audio_file):
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_file,
    ]
    try:
        result = subprocess.run(
            ffprobe_cmd, capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                return max(1, int(round(float(output) * 1000)))
    except (OSError, ValueError):
        pass

    if audio_file.lower().endswith(".wav"):
        try:
            with wave.open(audio_file, "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                if frame_rate > 0:
                    return max(1, int(round((frame_count / frame_rate) * 1000)))
        except wave.Error as exc:
            raise RuntimeError(f"WAV 时长解析失败: {audio_file}") from exc

    raise RuntimeError(
        "无法解析音频时长，请确认 ffprobe 可用，"
        f"或提供可读取时长的音频格式: {audio_file}"
    )


def _format_srt_timestamp(total_ms):
    total_ms = max(0, int(total_ms))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt_subtitles(audio_dir, script_lines, subtitle_path, pause_ms=400):
    segments = _collect_sorted_segments(audio_dir)
    if not segments:
        raise ValueError(f"未找到可用于字幕对齐的音频片段: {audio_dir}")

    subtitle_dir = os.path.dirname(os.path.abspath(subtitle_path))
    if subtitle_dir:
        os.makedirs(subtitle_dir, exist_ok=True)

    current_start_ms = 0
    with open(subtitle_path, "w", encoding="utf-8") as srt:
        for subtitle_index, segment_path in enumerate(segments, start=1):
            file_name = os.path.basename(segment_path)
            stem = os.path.splitext(file_name)[0]
            line_index = int(stem) if stem.isdigit() else subtitle_index - 1
            line_candidate = (
                script_lines[line_index] if line_index < len(script_lines) else {}
            )
            line = line_candidate if isinstance(line_candidate, dict) else {}

            speaker = str(line.get("speaker", "")).strip()
            text = str(line.get("text", "")).strip().replace("\n", " ")
            if speaker and text:
                subtitle_text = f"{speaker}：{text}"
            elif text:
                subtitle_text = text
            else:
                subtitle_text = f"[{stem}]"

            duration_ms = _probe_duration_ms(segment_path)
            end_ms = current_start_ms + duration_ms

            srt.write(f"{subtitle_index}\n")
            srt.write(
                f"{_format_srt_timestamp(current_start_ms)} --> "
                f"{_format_srt_timestamp(end_ms)}\n"
            )
            srt.write(f"{subtitle_text}\n\n")

            if subtitle_index < len(segments):
                current_start_ms = end_ms + pause_ms
            else:
                current_start_ms = end_ms

    return len(segments)


async def main():
    parser = argparse.ArgumentParser(description="Gnosis 有声书生产系统")
    parser.add_argument(
        "step",
        choices=["extract", "script", "tts", "merge", "full"],
        help="运行步骤: extract(选角), script(剧本), tts(语音), merge(混音), full(全流程)",
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
    parser.add_argument("--pause", type=int, default=400, help="句子间的停顿毫秒数")
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
        "--sovits-switch-url",
        default=DEFAULT_SWITCH_URL,
        help="GPT-SoVITS API 基地址（用于 /set_gpt_weights 与 /set_sovits_weights）",
    )
    parser.add_argument(
        "--subtitle",
        default="",
        help="输出字幕文件路径（SRT）；相对路径按项目目录解析，默认与最终音频同名",
    )

    args = parser.parse_args()
    try:
        project_name = validate_project_name(args.project)
    except ValueError as exc:
        parser.error(str(exc))

    project_root = os.path.abspath(os.path.join("data", "projects", project_name))
    os.makedirs(project_root, exist_ok=True)
    print(f"📁 当前 project: {project_name}")
    print(f"📂 project 目录: {project_root}")

    script_path = os.path.join(project_root, "script.json")
    audio_dir = os.path.join(project_root, "output_audio")
    character_db_path = os.path.join(project_root, "character_db.json")
    llm_cache_dir = resolve_project_path(project_root, args.llm_cache_dir)
    project_input_path = os.path.join(project_root, "input.txt")
    final_file = os.path.join(project_root, "final_audiobook.mp3")

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
            db_path=character_db_path, seeds_dir="voice/ref"
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
            text, char_manager, pass1_chunking_config, cache_dir=llm_cache_dir
        )  # 内部会自动 save_db
        print(f"✅ 角色库已更新: {len(char_manager.characters)} 个角色")

    # --- Step 2: 生成剧本 ---
    if args.step in ["script", "full"]:
        from gnosis.pipeline import run_pass2

        print("📝 [Step 2] 正在生成结构化剧本...")
        script_data = run_pass2(
            text, char_manager, pass2_chunking_config, cache_dir=llm_cache_dir
        )
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 剧本已保存至: {script_path}")

    # --- Step 3: 语音生成 (TTS) ---
    if args.step in ["tts", "full"]:
        print("🎙️ [Step 3] 正在调用 GPT-SoVITS 生成音频...")
        if not os.path.exists(script_path):
            print("❌ 错误: 找不到剧本文件，请先运行 script 步骤")
            return

        with open(script_path, "r", encoding="utf-8") as f:
            script_payload = json.load(f)
            characters, script_list = parse_script_payload(script_payload)

        speaker_to_voice, voice_specs, default_voice_id = build_voice_registry(
            characters, seeds_dir="voice/ref"
        )

        # 按声线分组，减少模型切换次数
        grouped_jobs = {}
        for i, line in enumerate(script_list):
            speaker = line["speaker"]
            voice_id = speaker_to_voice.get(speaker, default_voice_id)
            if voice_id not in voice_specs:
                voice_id = default_voice_id
            grouped_jobs.setdefault(voice_id, []).append((i, line))

        os.makedirs(audio_dir, exist_ok=True)
        for voice_id, jobs in grouped_jobs.items():
            voice_spec = voice_specs[voice_id]
            sovits_model_path = voice_spec.get("sovits_model_path")
            gpt_model_path = voice_spec.get("gpt_model_path")
            if sovits_model_path and gpt_model_path:
                switched = switch_character(
                    sovits_model_path,
                    gpt_model_path,
                    switch_url=args.sovits_switch_url,
                )
                if not switched:
                    raise RuntimeError(f"切换模型失败: voice={voice_id}")

            print(f"   声线分组: {voice_id}, 句子数={len(jobs)}")
            for i, line in jobs:
                file_path = os.path.join(audio_dir, f"{i:04d}.wav")
                if os.path.exists(file_path):
                    continue  # 跳过已存在的，方便断点续传

                print(f"   进度: {i + 1}/{len(script_list)} -> {line['speaker']}")
                ok = await tts_generate(
                    line["text"],
                    voice_spec,
                    file_path,
                    sovits_url=args.sovits_url,
                )
                if not ok:
                    raise RuntimeError(
                        f"TTS 调用失败: index={i}, speaker={line['speaker']}, voice={voice_id}"
                    )
        print("✅ 音频片段生成完毕")

    # --- Step 4: 合并混音 ---
    if args.step in ["merge", "full"]:
        import gnosis_rs

        print("🎚️ [Step 4] Rust 引擎正在混音并执行响度归一化...")
        if not os.path.exists(audio_dir):
            print("❌ 错误: 找不到音频目录，请先运行 tts 步骤")
            return

        success = gnosis_rs.merge_audio_pro(
            os.path.abspath(audio_dir), os.path.abspath(final_file), args.pause
        )

        if success:
            print(f"🎉 大功告成！最终成品: {final_file}")
            subtitle_file = (
                resolve_project_path(project_root, args.subtitle.strip())
                if args.subtitle.strip()
                else f"{os.path.splitext(final_file)[0]}.srt"
            )
            if os.path.exists(script_path):
                with open(script_path, "r", encoding="utf-8") as f:
                    script_payload = json.load(f)
                    _, script_list = parse_script_payload(script_payload)
                subtitle_count = generate_srt_subtitles(
                    audio_dir=os.path.abspath(audio_dir),
                    script_lines=script_list,
                    subtitle_path=os.path.abspath(subtitle_file),
                    pause_ms=args.pause,
                )
                print(
                    f"📝 已生成同步字幕: {subtitle_file} "
                    f"(共 {subtitle_count} 条，可直接导入剪辑软件)"
                )
            else:
                print("⚠️ 未找到剧本文件，跳过字幕生成")


if __name__ == "__main__":
    asyncio.run(main())
