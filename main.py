import argparse
import asyncio
import json
import os
import subprocess
import unicodedata
import wave
from gnosis.state_manager import CharacterManager
from gnosis.chunking import ChunkingConfig
from gnosis.project_prompt import ProjectPromptOverrides, load_project_prompt_overrides
from gnosis.tts_engine import (
    DEFAULT_COSYVOICE_MODE,
    DEFAULT_COSYVOICE_REF_DIR,
    DEFAULT_COSYVOICE_SAMPLE_RATE,
    DEFAULT_COSYVOICE_SPK_ID,
    DEFAULT_COSYVOICE_URL,
    DEFAULT_SOVITS_REF_DIR,
    DEFAULT_SOVITS_URL,
    DEFAULT_SWITCH_URL,
    DEFAULT_TTS_ENGINE,
    build_voice_registry,
    create_tts_engine,
)

DEFAULT_MIN_RATIO = 1200 / 1800
DEFAULT_MAX_RATIO = 2600 / 1800
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
PUNCTUATION_ONLY_SILENCE_MS = 400
DEFAULT_AUDIO_SAMPLE_RATE = 24000


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


def _filter_script_jobs_by_character(script_lines, character_name):
    target_character = (character_name or "").strip()
    jobs = []
    available_speakers = set()
    for i, line in enumerate(script_lines):
        if not isinstance(line, dict):
            continue
        speaker = str(line.get("speaker", "")).strip()
        if speaker:
            available_speakers.add(speaker)
        if not target_character or speaker == target_character:
            jobs.append((i, line))
    return jobs, available_speakers


def delete_character_audio_segments(audio_dir, script_lines, character_name):
    target_character = (character_name or "").strip()
    if not target_character:
        return {"target": "", "matched_lines": 0, "deleted_files": 0}

    target_indices = {
        i
        for i, line in enumerate(script_lines)
        if isinstance(line, dict) and str(line.get("speaker", "")).strip() == target_character
    }
    if not target_indices:
        return {
            "target": target_character,
            "matched_lines": 0,
            "deleted_files": 0,
        }
    if not os.path.isdir(audio_dir):
        return {
            "target": target_character,
            "matched_lines": len(target_indices),
            "deleted_files": 0,
        }

    deleted_files = 0
    for audio_path in _collect_sorted_segments(audio_dir):
        file_name = os.path.basename(audio_path)
        stem = os.path.splitext(file_name)[0]
        if not stem.isdigit():
            continue
        if int(stem) not in target_indices:
            continue
        os.remove(audio_path)
        deleted_files += 1

    return {
        "target": target_character,
        "matched_lines": len(target_indices),
        "deleted_files": deleted_files,
    }


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


def _pause_samples_from_ms(pause_ms, sample_rate):
    if pause_ms <= 0:
        return 0
    pause_samples = int(round(sample_rate * (pause_ms / 1000.0)))
    return max(1, pause_samples)


def _samples_to_ms(sample_index, sample_rate):
    return int(round((int(sample_index) * 1000.0) / int(sample_rate)))


def _build_precise_timeline(audio_dir, pause_ms):
    segments = _collect_sorted_segments(audio_dir)
    if not segments:
        raise ValueError(f"未找到可用于精确时间轴的音频片段: {audio_dir}")

    sample_rate = None
    channels = None
    timeline_segments = []
    running_sample = 0

    for i, segment_path in enumerate(segments):
        extension = os.path.splitext(segment_path)[1].lower()
        if extension != ".wav":
            raise ValueError(
                "精确字幕模式仅支持 WAV 分段，请先统一为 WAV: "
                f"{os.path.basename(segment_path)}"
            )

        try:
            with wave.open(segment_path, "rb") as wav_file:
                segment_sample_rate = wav_file.getframerate()
                segment_channels = wav_file.getnchannels()
                frame_count = wav_file.getnframes()
        except wave.Error as exc:
            raise RuntimeError(f"WAV 读取失败: {segment_path}") from exc

        if segment_sample_rate <= 0:
            raise ValueError(f"非法采样率: {segment_path}")
        if frame_count < 0:
            raise ValueError(f"非法帧数: {segment_path}")

        if sample_rate is None:
            sample_rate = int(segment_sample_rate)
            channels = int(segment_channels)
        else:
            if int(segment_sample_rate) != sample_rate:
                raise ValueError(
                    "精确字幕模式要求所有分段采样率一致: "
                    f"{os.path.basename(segment_path)}={segment_sample_rate}, "
                    f"expected={sample_rate}"
                )
            if int(segment_channels) != channels:
                raise ValueError(
                    "精确字幕模式要求所有分段声道一致: "
                    f"{os.path.basename(segment_path)}={segment_channels}, "
                    f"expected={channels}"
                )

        start_sample = running_sample
        end_sample = start_sample + int(frame_count)
        stem = os.path.splitext(os.path.basename(segment_path))[0]
        line_index = int(stem) if stem.isdigit() else i
        timeline_segments.append(
            {
                "file_name": os.path.basename(segment_path),
                "line_index": line_index,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration_samples": int(frame_count),
            }
        )
        running_sample = end_sample

    pause_samples = _pause_samples_from_ms(pause_ms, sample_rate)
    if pause_samples > 0 and len(timeline_segments) > 1:
        for i in range(1, len(timeline_segments)):
            offset = pause_samples * i
            timeline_segments[i]["start_sample"] += offset
            timeline_segments[i]["end_sample"] += offset
        running_sample += pause_samples * (len(timeline_segments) - 1)

    return {
        "audio_dir": os.path.abspath(audio_dir),
        "sample_rate": sample_rate,
        "channels": channels,
        "pause_ms": int(pause_ms),
        "pause_samples": pause_samples,
        "segment_count": len(timeline_segments),
        "total_samples": running_sample,
        "segments": timeline_segments,
    }


def _write_timeline_file(timeline_path, timeline_payload):
    parent = os.path.dirname(os.path.abspath(timeline_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, ensure_ascii=False, indent=2)


def _read_wav_samples(wav_path):
    try:
        with wave.open(wav_path, "rb") as wav_file:
            return int(wav_file.getnframes()), int(wav_file.getframerate())
    except wave.Error as exc:
        raise RuntimeError(f"WAV 读取失败: {wav_path}") from exc


def _is_punctuation_only_text(text):
    value = (text or "").strip()
    if not value:
        return True

    has_punctuation = False
    for ch in value:
        if ch.isspace():
            continue
        if unicodedata.category(ch).startswith("P"):
            has_punctuation = True
            continue
        return False
    return has_punctuation


def _write_silence_wav(output_path, duration_ms, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    frame_count = max(1, int(round(sample_rate * (duration_ms / 1000.0))))
    silence_bytes = b"\x00\x00" * frame_count  # mono int16
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(silence_bytes)


def _format_srt_timestamp(total_ms):
    total_ms = max(0, int(total_ms))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _infer_effective_pause_ms(
    configured_pause_ms, segment_durations_ms, final_audio_path=None
):
    segment_count = len(segment_durations_ms)
    if segment_count <= 1:
        return 0, None

    if not final_audio_path or not os.path.exists(final_audio_path):
        return configured_pause_ms, None

    final_audio_ms = _probe_duration_ms(final_audio_path)
    total_voice_ms = sum(segment_durations_ms)
    inferred_pause = (final_audio_ms - total_voice_ms) / (segment_count - 1)
    if inferred_pause < 0:
        return configured_pause_ms, final_audio_ms
    return int(round(inferred_pause)), final_audio_ms


def generate_srt_subtitles_precise(script_lines, subtitle_path, timeline_payload):
    segments = timeline_payload.get("segments") or []
    if not segments:
        raise ValueError("时间轴为空，无法生成字幕")

    sample_rate = int(timeline_payload.get("sample_rate") or 0)
    if sample_rate <= 0:
        raise ValueError("时间轴采样率无效，无法生成字幕")

    subtitle_dir = os.path.dirname(os.path.abspath(subtitle_path))
    if subtitle_dir:
        os.makedirs(subtitle_dir, exist_ok=True)

    last_written_end_ms = 0
    with open(subtitle_path, "w", encoding="utf-8") as srt:
        for subtitle_index, segment in enumerate(segments, start=1):
            line_index = int(segment.get("line_index", subtitle_index - 1))
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
                subtitle_text = f"[{segment.get('file_name', subtitle_index)}]"

            start_ms = _samples_to_ms(segment.get("start_sample", 0), sample_rate)
            end_ms = _samples_to_ms(segment.get("end_sample", 0), sample_rate)
            if start_ms < last_written_end_ms:
                start_ms = last_written_end_ms
            if end_ms <= start_ms:
                end_ms = start_ms + 1

            srt.write(f"{subtitle_index}\n")
            srt.write(
                f"{_format_srt_timestamp(start_ms)} --> "
                f"{_format_srt_timestamp(end_ms)}\n"
            )
            srt.write(f"{subtitle_text}\n\n")
            last_written_end_ms = end_ms

    return {
        "subtitle_count": len(segments),
        "sample_rate": sample_rate,
        "pause_ms": int(timeline_payload.get("pause_ms", 0)),
        "pause_samples": int(timeline_payload.get("pause_samples", 0)),
        "total_samples": int(timeline_payload.get("total_samples", 0)),
        "predicted_total_ms": _samples_to_ms(
            int(timeline_payload.get("total_samples", 0)), sample_rate
        ),
    }


def generate_srt_subtitles(
    audio_dir,
    script_lines,
    subtitle_path,
    pause_ms=400,
    final_audio_path=None,
):
    segments = _collect_sorted_segments(audio_dir)
    if not segments:
        raise ValueError(f"未找到可用于字幕对齐的音频片段: {audio_dir}")

    subtitle_dir = os.path.dirname(os.path.abspath(subtitle_path))
    if subtitle_dir:
        os.makedirs(subtitle_dir, exist_ok=True)

    segment_durations_ms = [_probe_duration_ms(path) for path in segments]
    effective_pause_ms, final_audio_ms = _infer_effective_pause_ms(
        pause_ms, segment_durations_ms, final_audio_path=final_audio_path
    )

    predicted_total_ms = sum(segment_durations_ms)
    if len(segments) > 1:
        predicted_total_ms += effective_pause_ms * (len(segments) - 1)

    scale = 1.0
    if final_audio_ms and predicted_total_ms > 0:
        scale = final_audio_ms / predicted_total_ms

    current_start_raw_ms = 0
    last_written_end_ms = 0
    with open(subtitle_path, "w", encoding="utf-8") as srt:
        for subtitle_index, (segment_path, duration_ms) in enumerate(
            zip(segments, segment_durations_ms), start=1
        ):
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

            current_end_raw_ms = current_start_raw_ms + duration_ms
            start_ms = int(round(current_start_raw_ms * scale))
            end_ms = int(round(current_end_raw_ms * scale))
            if subtitle_index == len(segments) and final_audio_ms is not None:
                end_ms = final_audio_ms
            if start_ms < last_written_end_ms:
                start_ms = last_written_end_ms
            if end_ms <= start_ms:
                end_ms = start_ms + 1

            srt.write(f"{subtitle_index}\n")
            srt.write(
                f"{_format_srt_timestamp(start_ms)} --> "
                f"{_format_srt_timestamp(end_ms)}\n"
            )
            srt.write(f"{subtitle_text}\n\n")
            last_written_end_ms = end_ms

            if subtitle_index < len(segments):
                current_start_raw_ms = current_end_raw_ms + effective_pause_ms
            else:
                current_start_raw_ms = current_end_raw_ms

    return {
        "subtitle_count": len(segments),
        "configured_pause_ms": pause_ms,
        "effective_pause_ms": effective_pause_ms,
        "final_audio_ms": final_audio_ms,
        "predicted_total_ms": predicted_total_ms,
        "scale": scale,
    }


async def run_cosyvoice_concurrent_tts(
    tts_engine,
    grouped_jobs,
    voice_specs,
    audio_dir,
    total_lines,
    workers,
    sample_rate,
):
    pending_jobs = []
    for voice_id, jobs in grouped_jobs.items():
        voice_spec = voice_specs[voice_id]
        for i, line in jobs:
            file_path = os.path.join(audio_dir, f"{i:04d}.wav")
            if os.path.exists(file_path):
                continue
            pending_jobs.append((i, line, voice_id, voice_spec, file_path))

    if not pending_jobs:
        print("   所有音频片段已存在，跳过生成")
        return

    pending_jobs.sort(key=lambda item: item[0])
    job_queue = asyncio.Queue()
    for job in pending_jobs:
        job_queue.put_nowait(job)

    max_workers = max(1, min(workers, len(pending_jobs)))
    print(f"   CosyVoice 并发线程: {max_workers}, 待生成={len(pending_jobs)}")
    progress_lock = asyncio.Lock()
    state = {"done": 0, "error": None}

    async def worker():
        while True:
            if state["error"] is not None:
                return
            try:
                i, line, voice_id, voice_spec, file_path = job_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                text = line.get("text", "")
                emotion = line.get("emotion", "")
                normalized_text = text if isinstance(text, str) else ""
                if _is_punctuation_only_text(normalized_text):
                    _write_silence_wav(
                        file_path,
                        duration_ms=PUNCTUATION_ONLY_SILENCE_MS,
                        sample_rate=sample_rate,
                    )
                    ok = True
                else:
                    ok = await tts_engine.generate_line(
                        text=normalized_text,
                        emotion=emotion if isinstance(emotion, str) else "",
                        voice_id=voice_id,
                        voice_spec=voice_spec,
                        output_path=file_path,
                    )
                if not ok:
                    raise RuntimeError(
                        "TTS 调用失败:"
                        f" engine={tts_engine.name}, index={i},"
                        f" speaker={line['speaker']}, voice={voice_id}, text={text}"
                    )

                async with progress_lock:
                    state["done"] += 1
                    print(
                        "   进度:"
                        f" {state['done']}/{len(pending_jobs)}"
                        f" -> {line['speaker']} (line {i + 1}/{total_lines})"
                    )
            except Exception as exc:
                async with progress_lock:
                    if state["error"] is None:
                        state["error"] = exc
            finally:
                job_queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(max_workers)]
    await asyncio.gather(*tasks)
    if state["error"] is not None:
        raise state["error"]


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
        "--sovits-switch-url",
        default=DEFAULT_SWITCH_URL,
        help="GPT-SoVITS API 基地址（用于 /set_gpt_weights 与 /set_sovits_weights）",
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
        "--cosyvoice-url",
        default=DEFAULT_COSYVOICE_URL,
        help="CosyVoice FastAPI 地址（可传主机地址或具体 inference 端点）",
    )
    parser.add_argument(
        "--cosyvoice-mode",
        default=DEFAULT_COSYVOICE_MODE,
        choices=["instruct", "instruct2"],
        help="CosyVoice 模式：instruct(预置说话人) / instruct2(参考音频)",
    )
    parser.add_argument(
        "--cosyvoice-sample-rate",
        type=int,
        default=DEFAULT_COSYVOICE_SAMPLE_RATE,
        help="CosyVoice 输出采样率（用于将 API 返回 PCM 包装为 wav）",
    )
    parser.add_argument(
        "--cosyvoice-spk-id",
        default=DEFAULT_COSYVOICE_SPK_ID,
        help="CosyVoice instruct 模式默认 spk_id（默认 logos；优先级低于 .ref 中 cosyvoice_spk_id）",
    )
    parser.add_argument(
        "--subtitle",
        default="",
        help="输出字幕文件路径（SRT）；相对路径按项目目录解析，默认与最终音频同名",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="proofread Web 服务监听地址",
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

    try:
        project_name = validate_project_name(args.project)
    except ValueError as exc:
        parser.error(str(exc))

    project_root = os.path.abspath(os.path.join("data", "projects", project_name))
    os.makedirs(project_root, exist_ok=True)
    print(f"📁 当前 project: {project_name}")
    print(f"📂 project 目录: {project_root}")

    if args.step == "proofread":
        from gnosis.proofread_web import run_proofread_server

        run_proofread_server(
            project_name=project_name,
            host=args.web_host,
            port=args.web_port,
            projects_root=os.path.join("data", "projects"),
        )
        return

    script_path = os.path.join(project_root, "script.json")
    audio_dir = os.path.join(project_root, "output_audio")
    character_db_path = os.path.join(project_root, "character_db.json")
    llm_cache_dir = resolve_project_path(project_root, args.llm_cache_dir)
    project_input_path = os.path.join(project_root, "input.txt")
    final_file = os.path.join(project_root, "final_audiobook.mp3")
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
        character_seeds_dir = (
            DEFAULT_COSYVOICE_REF_DIR
            if args.tts_engine == "cosyvoice"
            else DEFAULT_SOVITS_REF_DIR
        )
        char_manager = CharacterManager(
            db_path=character_db_path, seeds_dir=character_seeds_dir
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
        char_target = args.char.strip()
        delete_only_mode = args.step == "tts" and bool(delete_target) and not bool(char_target)
        if delete_only_mode:
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

        tts_engine = create_tts_engine(
            engine_name=args.tts_engine,
            sovits_url=args.sovits_url,
            sovits_switch_url=args.sovits_switch_url,
            cosyvoice_url=args.cosyvoice_url,
            cosyvoice_mode=args.cosyvoice_mode,
            cosyvoice_sample_rate=args.cosyvoice_sample_rate,
            cosyvoice_spk_id=args.cosyvoice_spk_id,
        )
        segment_sample_rate = DEFAULT_AUDIO_SAMPLE_RATE
        if args.tts_engine == "cosyvoice":
            cosyvoice_model = getattr(tts_engine, "cosyvoice", None)
            model_rate = getattr(cosyvoice_model, "sample_rate", None)
            if model_rate:
                segment_sample_rate = int(model_rate)
        tts_jobs, available_speakers = _filter_script_jobs_by_character(
            script_list, args.char
        )
        if args.char.strip():
            if not tts_jobs:
                suggestion = "、".join(sorted(available_speakers)[:20])
                if suggestion:
                    raise ValueError(
                        f"--char 未命中剧中人物: {args.char.strip()}。可用人物示例: {suggestion}"
                    )
                raise ValueError(
                    f"--char 未命中剧中人物: {args.char.strip()}（剧本中未找到可用 speaker）"
                )
            print(
                "   角色过滤:"
                f" 仅生成 {args.char.strip()}，命中 {len(tts_jobs)}/{len(script_list)} 句"
            )

        tts_ref_dir = (
            DEFAULT_COSYVOICE_REF_DIR
            if args.tts_engine == "cosyvoice"
            else DEFAULT_SOVITS_REF_DIR
        )
        print(f"   ref目录: {os.path.abspath(tts_ref_dir)}")
        speaker_to_voice, voice_specs, default_voice_id = build_voice_registry(
            characters,
            seeds_dir=tts_ref_dir,
            ref_format=args.tts_engine,
        )

        # 按声线分组，减少模型切换次数
        grouped_jobs = {}
        for i, line in tts_jobs:
            speaker = line["speaker"]
            voice_id = speaker_to_voice.get(speaker, default_voice_id)
            if voice_id not in voice_specs:
                voice_id = default_voice_id
            grouped_jobs.setdefault(voice_id, []).append((i, line))

        os.makedirs(audio_dir, exist_ok=True)
        for voice_id, jobs in grouped_jobs.items():
            print(f"   声线分组: {voice_id}, 句子数={len(jobs)}")

        if args.tts_engine == "cosyvoice":
            for voice_id in grouped_jobs:
                voice_spec = voice_specs[voice_id]
                prepared = tts_engine.prepare_voice(voice_id, voice_spec)
                if not prepared:
                    raise RuntimeError(
                        f"TTS 声线准备失败: engine={args.tts_engine}, voice={voice_id}"
                    )
            await run_cosyvoice_concurrent_tts(
                tts_engine=tts_engine,
                grouped_jobs=grouped_jobs,
                voice_specs=voice_specs,
                audio_dir=audio_dir,
                total_lines=len(script_list),
                workers=args.tts_workers,
                sample_rate=segment_sample_rate,
            )
        else:
            for voice_id, jobs in grouped_jobs.items():
                voice_spec = voice_specs[voice_id]
                prepared = tts_engine.prepare_voice(voice_id, voice_spec)
                if not prepared:
                    raise RuntimeError(
                        f"TTS 声线准备失败: engine={args.tts_engine}, voice={voice_id}"
                    )

                for i, line in jobs:
                    file_path = os.path.join(audio_dir, f"{i:04d}.wav")
                    if os.path.exists(file_path):
                        continue  # 跳过已存在的，方便断点续传

                    print(f"   进度: {i + 1}/{len(script_list)} -> {line['speaker']}")
                    text = line.get("text", "")
                    emotion = line.get("emotion", "")
                    normalized_text = text if isinstance(text, str) else ""
                    if _is_punctuation_only_text(normalized_text):
                        _write_silence_wav(
                            file_path,
                            duration_ms=PUNCTUATION_ONLY_SILENCE_MS,
                            sample_rate=segment_sample_rate,
                        )
                        ok = True
                    else:
                        ok = await tts_engine.generate_line(
                            text=normalized_text,
                            emotion=emotion if isinstance(emotion, str) else "",
                            voice_id=voice_id,
                            voice_spec=voice_spec,
                            output_path=file_path,
                        )
                    if not ok:
                        raise RuntimeError(
                            "TTS 调用失败:"
                            f" engine={args.tts_engine}, index={i},"
                            f" speaker={line['speaker']}, voice={voice_id}, text={text}"
                        )
        if args.delete_char_audio.strip():
            delete_stats = delete_character_audio_segments(
                audio_dir=audio_dir,
                script_lines=script_list,
                character_name=args.delete_char_audio,
            )
            if delete_stats["matched_lines"] == 0:
                print(
                    "⚠️ 删除阶段未命中角色:"
                    f" {args.delete_char_audio.strip()}（未删除任何文件）"
                )
            else:
                print(
                    "🧹 删除角色语音完成:"
                    f" {delete_stats['target']}，命中台词 {delete_stats['matched_lines']} 句，"
                    f" 删除文件 {delete_stats['deleted_files']} 个"
                )
        print("✅ 音频片段生成完毕")

    # --- Step 4: 合并混音 ---
    if args.step in ["merge", "full"]:
        import gnosis_rs

        print("🎚️ [Step 4] Rust 引擎正在混音并执行响度归一化...")
        if not os.path.exists(audio_dir):
            print("❌ 错误: 找不到音频目录，请先运行 tts 步骤")
            return

        precise_timeline = _build_precise_timeline(os.path.abspath(audio_dir), args.pause)
        timeline_file = f"{os.path.splitext(final_file)[0]}.timeline.json"
        _write_timeline_file(os.path.abspath(timeline_file), precise_timeline)
        print(
            "🧭 已生成样本级时间轴:"
            f" {timeline_file} (sr={precise_timeline['sample_rate']}Hz, "
            f"pause={precise_timeline['pause_samples']} samples)"
        )

        success = gnosis_rs.merge_audio_pro(
            input_dir=os.path.abspath(audio_dir),
            output_file=os.path.abspath(final_file),
            pause_ms=args.pause,
            silence_sample_rate=precise_timeline["sample_rate"],
            keep_master=True,
        )

        if success:
            master_file = f"{os.path.splitext(final_file)[0]}.master.wav"
            if os.path.exists(master_file):
                master_samples, master_sample_rate = _read_wav_samples(master_file)
                if master_sample_rate != precise_timeline["sample_rate"]:
                    raise RuntimeError(
                        "主母带采样率与时间轴不一致: "
                        f"master={master_sample_rate}, "
                        f"timeline={precise_timeline['sample_rate']}"
                    )
                if master_samples != precise_timeline["total_samples"]:
                    raise RuntimeError(
                        "主母带总样本数与理论值不一致: "
                        f"master={master_samples}, "
                        f"timeline={precise_timeline['total_samples']}"
                    )
            else:
                raise RuntimeError(f"未找到主母带文件: {master_file}")

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
