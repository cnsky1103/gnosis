from typing import Dict
from urllib.parse import urlsplit, urlunsplit

import requests
import os

from gnosis.tts.tts_engine import BaseTTSEngine
from gnosis.tts.tts_utils import DEFAULT_SOVITS_URL, is_han_char


HTTP = requests.Session()
HTTP.trust_env = False

DEFAULT_VOICE_ID = "logos"
GPT_WEIGHTS_DIR = "GPT_weights"
SOVITS_WEIGHTS_DIR = "SoVITS_weights"
DEFAULT_SOVITS_REF_DIR = "voice/ref/sovits"

def _normalize_model_path(raw_model_path, model_dir):
    if not raw_model_path:
        return None
    normalized = raw_model_path.strip().replace("\\", "/")
    if not normalized:
        return None
    if normalized.startswith(f"{model_dir}/"):
        return normalized
    if os.path.isabs(normalized):
        return normalized
    normalized = normalized.lstrip("./")
    return f"{model_dir}/{normalized}"



def _resolve_path(base_dir, raw_path):
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(base_dir, raw_path))

class GptSoVitsEngine(BaseTTSEngine):
    name = "gpt-sovits"

    def __init__(
        self,
        base_url: str = DEFAULT_SOVITS_URL
    ):
        self.base_url = base_url
        self.sovits_url = self.base_url + '/tts'
        self.switch_url =self.base_url 
        self.default_voice_id = "logos"
    
    def init(self, characters):
        build_voice_registry(characters)

    def prepare_voice(self, voice_id: str, voice_spec: Dict) -> bool:
        sovits_model_path = voice_spec.get("sovits_model_path")
        gpt_model_path = voice_spec.get("gpt_model_path")
        if sovits_model_path and gpt_model_path:
            return switch_character(
                sovits_model_path,
                gpt_model_path,
                switch_url=self.switch_url,
            )
        return True

    async def generate_line(
        self,
        text: str,
        output_path: str,
    ) -> bool:
        #emotion: Optional[str],
        #voice_id: str,
        #voice_spec: Dict,
        endpoint = self._sovits_tts_endpoint(self.sovits_url)
        payload = {
            "text": text,
            "text_lang": self._resolve_text_lang(text),
            "ref_audio_path": os.path.abspath(voice_spec["ref_audio_path"]),
            "prompt_text": voice_spec["prompt_text"],
            "prompt_lang": "zh",
            "text_split_method": "cut5",
            "batch_size": 1,
            "streaming_mode": False,
            "speed_factor": 1.1,
        }

        try:
            response = HTTP.post(endpoint, json=payload, timeout=120)
        except requests.RequestException:
            return False

        if response.status_code != 200:
            print(f"   /tts 失败({response.status_code}): {response.text[:160]}")
            return False

        with open(output_path, "wb") as f:
            f.write(response.content)
        return True


    def switch_character(self,sovits_path, gpt_path):
        # api_v2: /set_gpt_weights + /set_sovits_weights
        gpt_endpoint = f"{self.base_url}/set_gpt_weights"
        sovits_endpoint = f"{self.base_url}/set_sovits_weights"

        try:
            gpt_resp = HTTP.get(gpt_endpoint, params={"weights_path": gpt_path}, timeout=30)
            sovits_resp = HTTP.get(
                sovits_endpoint, params={"weights_path": sovits_path}, timeout=30
            )
        except requests.RequestException as exc:
            print(f"   模型切换异常: {exc}")
            return False

        if gpt_resp.status_code == 200 and sovits_resp.status_code == 200:
            print(f"   模型切换成功: sovits={sovits_path}")
            return True

        print(
            "   模型切换失败:"
            f" gpt={gpt_resp.status_code}, sovits={sovits_resp.status_code},"
            f" gpt_body={gpt_resp.text[:120]}, sovits_body={sovits_resp.text[:120]}"
        )
        return False


    def _sovits_tts_endpoint(url_or_endpoint):
        parsed = urlsplit(url_or_endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"无效URL: {url_or_endpoint}")
        if parsed.path.rstrip("/") == "/tts":
            return url_or_endpoint
        return urlunsplit((parsed.scheme, parsed.netloc, "/tts", "", ""))

    
    def _resolve_text_lang(text):
        # 含有非汉字字母（如英文、日文假名等）时走 auto，纯中文默认 zh
        for ch in text:
            if ch.isalpha() and not is_han_char(ch):
                return "auto"
        return "zh"
    
    def build_voice_registry(
        characters,
        seeds_dir=DEFAULT_SOVITS_REF_DIR,
        ref_format="sovits",
    ):
        # 从 script.json 的 characters 构建:
        # 1) speaker -> voice_id
        # 2) voice_id -> voice_spec
        speaker_to_voice = {}
        voice_specs = {}
        loader = _resolve_ref_loader(ref_format)
        cosyvoice_format = _is_cosyvoice_ref_format(ref_format)

        for char in characters:
            if not isinstance(char, dict):
                continue
            name = char.get("name")
            if not isinstance(name, str) or not name.strip():
                continue

            voice_id = char.get("voice") or DEFAULT_VOICE_ID
            speaker_to_voice[name] = voice_id

            if voice_id in voice_specs:
                continue

            ref_file = _build_ref_file_path(seeds_dir, voice_id)
            spec = loader(ref_file)
            if not spec and cosyvoice_format:
                spec = _build_default_cosyvoice_voice_spec()
            if not spec:
                continue
            spec["voice_id"] = voice_id
            voice_specs[voice_id] = spec

        default_ref_file = _build_ref_file_path(seeds_dir, DEFAULT_VOICE_ID)
        default_spec = loader(default_ref_file)
        if not default_spec and cosyvoice_format:
            default_spec = _build_default_cosyvoice_voice_spec(
                default_spk_id=DEFAULT_COSYVOICE_SPK_ID
            )
        if not default_spec:
            raise FileNotFoundError(
                f"默认声线ref不存在或格式错误: {default_ref_file} (format={ref_format})"
            )
        default_spec["voice_id"] = DEFAULT_VOICE_ID

        if DEFAULT_VOICE_ID not in voice_specs:
            voice_specs[DEFAULT_VOICE_ID] = default_spec

        return speaker_to_voice, voice_specs, DEFAULT_VOICE_ID

    def generate_script_tts(self, script_path, audio_dir, char=None, **kargs):
        #     for voice_id, jobs in grouped_jobs.items():
        #         voice_spec = voice_specs[voice_id]
        #         prepared = tts_engine.prepare_voice(voice_id, voice_spec)
        #         if not prepared:
        #             raise RuntimeError(
        #                 f"TTS 声线准备失败: engine={args.tts_engine}, voice={voice_id}"
        #             )

        #         for i, line in jobs:
        #             file_path = os.path.join(audio_dir, f"{i:04d}.wav")
        #             if os.path.exists(file_path):
        #                 continue  # 跳过已存在的，方便断点续传

        #             print(f"   进度: {i + 1}/{len(script_list)} -> {line['speaker']}")
        #             text = line.get("text", "")
        #             emotion = line.get("emotion", "")
        #             normalized_text = text if isinstance(text, str) else ""
        #             if is_punctuation_only_text(normalized_text):
        #                 _write_silence_wav(
        #                     file_path,
        #                     duration_ms=PUNCTUATION_ONLY_SILENCE_MS,
        #                     sample_rate=segment_sample_rate,
        #                 )
        #                 ok = True
        #             else:
        #                 ok = await tts_engine.generate_line(
        #                     text=normalized_text,
        #                     emotion=emotion if isinstance(emotion, str) else "",
        #                     voice_id=voice_id,
        #                     voice_spec=voice_spec,
        #                     output_path=file_path,
        #                 )
        #             if not ok:
        #                 raise RuntimeError(
        #                     "TTS 调用失败:"
        #                     f" engine={args.tts_engine}, index={i},"
        #                     f" speaker={line['speaker']}, voice={voice_id}, text={text}"
        #                 )
        return super().generate_script_tts(script_path, audio_dir, char, **kargs)


def _load_sovits_voice_spec(ref_file):
    lines = _read_ref_lines(ref_file)
    if not lines:
        return None

    base_dir = os.path.dirname(ref_file)
    gpt_model_path = None
    sovits_model_path = None
    ref_audio_path = None
    prompt_lines = []
    cosyvoice_spk_id = None

    # .ref 标准格式:
    # 1 gpt_model_path, 2 sovits_model_path, 3 ref_audio_path, 4+ prompt_text
    if len(lines) >= 4:
        gpt_model_path = _normalize_model_path(lines[0], GPT_WEIGHTS_DIR)
        sovits_model_path = _normalize_model_path(lines[1], SOVITS_WEIGHTS_DIR)
        ref_audio_path = _resolve_path(base_dir, lines[2])
        prompt_lines = lines[3:]
    # 兼容最简格式: 1 ref_audio_path, 2+ prompt_text
    elif len(lines) >= 2:
        ref_audio_path = _resolve_path(base_dir, lines[0])
        prompt_lines = lines[1:]
    else:
        ref_audio_path = _resolve_path(base_dir, lines[0])

    extracted_prompt_lines = []
    for raw_line in prompt_lines:
        lowered = raw_line.lower()
        if lowered.startswith("cosyvoice_spk_id="):
            cosyvoice_spk_id = raw_line.split("=", 1)[1].strip()
            continue
        if lowered.startswith("cosyvoice_spk_id:"):
            cosyvoice_spk_id = raw_line.split(":", 1)[1].strip()
            continue
        extracted_prompt_lines.append(raw_line)

    if not ref_audio_path:
        return None

    return {
        "ref_format": "sovits",
        "gpt_model_path": gpt_model_path,
        "sovits_model_path": sovits_model_path,
        "ref_audio_path": ref_audio_path,
        "prompt_text": " ".join(extracted_prompt_lines)
        or "这是系统默认旁白的参考文本。",
        "cosyvoice_spk_id": cosyvoice_spk_id,
    }


def _build_ref_file_path(seeds_dir, voice_id):
    return os.path.join(seeds_dir, f"{voice_id}.ref")


def _read_ref_lines(ref_file):
    if not os.path.exists(ref_file):
        return None

    with open(ref_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            lines.append(stripped)

    if not lines:
        return None
    return lines


def _split_ref_kv(raw_line):
    for separator in ("=", ":"):
        if separator not in raw_line:
            continue
        key, value = raw_line.split(separator, 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            return key, value
    return None, None


def _looks_like_wav(binary: bytes) -> bool:
    return len(binary) >= 12 and binary[:4] == b"RIFF" and binary[8:12] == b"WAVE"


def _write_pcm16_to_wav(output_path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm_bytes)

