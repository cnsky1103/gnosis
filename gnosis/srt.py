import os
import soundfile as sf
import wave
import json

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

def _samples_to_ms(sample_index, sample_rate):
    return int(round((int(sample_index) * 1000.0) / int(sample_rate)))

def _format_srt_timestamp(total_ms):
    total_ms = max(0, int(total_ms))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

from gnosis.utils import collect_sorted_segments

def _read_wav_samples(wav_path):
    try:
        with wave.open(wav_path, "rb") as wav_file:
            return int(wav_file.getnframes()), int(wav_file.getframerate())
    except wave.Error as exc:
        raise RuntimeError(f"WAV 读取失败: {wav_path}") from exc


def check_sample_rate(final_file, precise_timeline):
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

def build_precise_timeline(audio_dir, pause_ms):
    segments = collect_sorted_segments(audio_dir)
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
            with sf.SoundFile(segment_path) as f:
                segment_sample_rate = f.samplerate
                segment_channels = f.channels
                frame_count = f.frames
        except:
            raise RuntimeError(f"WAV 读取失败: {segment_path}")

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


def write_timeline_file(timeline_path, timeline_payload):
    parent = os.path.dirname(os.path.abspath(timeline_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, ensure_ascii=False, indent=2)

def _pause_samples_from_ms(pause_ms, sample_rate):
    if pause_ms <= 0:
        return 0
    pause_samples = int(round(sample_rate * (pause_ms / 1000.0)))
    return max(1, pause_samples)