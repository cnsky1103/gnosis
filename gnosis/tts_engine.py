import asyncio
import os
import wave
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

import requests

DEFAULT_TTS_ENGINE = "cosyvoice"
SUPPORTED_TTS_ENGINES = ("cosyvoice", "gpt-sovits")

DEFAULT_SOVITS_URL = "http://127.0.0.1:9880/tts"
DEFAULT_VOICE_ID = "logos"
DEFAULT_SWITCH_URL = "http://127.0.0.1:9880"
GPT_WEIGHTS_DIR = "GPT_weights"
SOVITS_WEIGHTS_DIR = "SoVITS_weights"
DEFAULT_SOVITS_REF_DIR = "voice/ref/sovits"

DEFAULT_COSYVOICE_URL = "http://127.0.0.1:50000"
DEFAULT_COSYVOICE_MODE = "instruct"
DEFAULT_COSYVOICE_SAMPLE_RATE = 22050
DEFAULT_COSYVOICE_REF_DIR = "voice/ref/cosyvoice"
DEFAULT_COSYVOICE_SPK_ID = DEFAULT_VOICE_ID
DEFAULT_COSYVOICE_MODEL_DIR = (
    "/Users/sky/code/CosyVoice/pretrained_models/CosyVoice2-0.5B"
)

HTTP = requests.Session()
HTTP.trust_env = False


class BaseTTSEngine:
    name = "base"

    def prepare_voice(self, voice_id: str, voice_spec: Dict) -> bool:
        return True

    async def generate_line(
        self,
        text: str,
        emotion: Optional[str],
        voice_id: str,
        voice_spec: Dict,
        output_path: str,
    ) -> bool:
        raise NotImplementedError


def _is_han_char(ch):
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
    )


def _resolve_text_lang(text):
    # 含有非汉字字母（如英文、日文假名等）时走 auto，纯中文默认 zh
    for ch in text:
        if ch.isalpha() and not _is_han_char(ch):
            return "auto"
    return "zh"


def _resolve_path(base_dir, raw_path):
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(base_dir, raw_path))


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


def _base_url(url_or_endpoint):
    parsed = urlsplit(url_or_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效URL: {url_or_endpoint}")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _sovits_tts_endpoint(url_or_endpoint):
    parsed = urlsplit(url_or_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效URL: {url_or_endpoint}")
    if parsed.path.rstrip("/") == "/tts":
        return url_or_endpoint
    return urlunsplit((parsed.scheme, parsed.netloc, "/tts", "", ""))


def _cosyvoice_endpoint(url_or_endpoint, mode):
    parsed = urlsplit(url_or_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效URL: {url_or_endpoint}")

    path = parsed.path.rstrip("/")
    if path.startswith("/inference_"):
        return url_or_endpoint

    return urlunsplit((parsed.scheme, parsed.netloc, f"/inference_{mode}", "", ""))


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


def _looks_like_audio_path(raw_value):
    normalized = str(raw_value or "").strip().lower()
    if not normalized:
        return False
    return normalized.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"))


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


def _load_cosyvoice_voice_spec(ref_file):
    lines = _read_ref_lines(ref_file)
    if not lines:
        return None

    base_dir = os.path.dirname(ref_file)
    ref_audio_path = None
    cosyvoice_spk_id = None
    prompt_lines = []
    fallback_lines = []

    for raw_line in lines:
        key, value = _split_ref_kv(raw_line)
        if key in {"ref_audio_path", "ref_audio", "ref_wav", "prompt_wav", "wav"}:
            ref_audio_path = _resolve_path(base_dir, value)
            continue
        if key in {"cosyvoice_spk_id", "spk_id", "speaker_id"}:
            cosyvoice_spk_id = value
            continue
        if key in {"prompt_text", "prompt", "text"}:
            prompt_lines.append(value)
            continue
        fallback_lines.append(raw_line)

    if not ref_audio_path and fallback_lines:
        first_line = fallback_lines[0]
        if _looks_like_audio_path(first_line):
            ref_audio_path = _resolve_path(base_dir, first_line)
            fallback_lines = fallback_lines[1:]

    if not prompt_lines and fallback_lines:
        prompt_lines = fallback_lines

    if not ref_audio_path and not cosyvoice_spk_id:
        return None

    return {
        "ref_format": "cosyvoice",
        "gpt_model_path": None,
        "sovits_model_path": None,
        "ref_audio_path": ref_audio_path,
        "prompt_text": " ".join(prompt_lines) or "这是系统默认旁白的参考文本。",
        "cosyvoice_spk_id": cosyvoice_spk_id,
    }


def _resolve_ref_loader(ref_format):
    normalized = (ref_format or "").strip().lower()
    if normalized in {"sovits", "gpt-sovits", "gpt_sovits"}:
        return _load_sovits_voice_spec
    if normalized in {"cosyvoice", "cosy"}:
        return _load_cosyvoice_voice_spec
    raise ValueError(f"不支持的 ref 格式: {ref_format}")


def _is_cosyvoice_ref_format(ref_format):
    normalized = (ref_format or "").strip().lower()
    return normalized in {"cosyvoice", "cosy"}


def _build_default_cosyvoice_voice_spec(default_spk_id=None):
    return {
        "ref_format": "cosyvoice",
        "gpt_model_path": None,
        "sovits_model_path": None,
        "ref_audio_path": None,
        "prompt_text": "这是系统默认旁白的参考文本。",
        "cosyvoice_spk_id": (default_spk_id or "").strip() or None,
    }


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


def switch_character(sovits_path, gpt_path, switch_url=DEFAULT_SWITCH_URL):
    # api_v2: /set_gpt_weights + /set_sovits_weights
    base = _base_url(switch_url)
    gpt_endpoint = f"{base}/set_gpt_weights"
    sovits_endpoint = f"{base}/set_sovits_weights"

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


def _build_cosyvoice_instruct_text(emotion: Optional[str]) -> str:
    emotion_text = (emotion or "").strip() or "neutral"
    return f"这句话的情感是：{emotion_text}"


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


class GptSoVitsEngine(BaseTTSEngine):
    name = "gpt-sovits"

    def __init__(
        self,
        sovits_url: str = DEFAULT_SOVITS_URL,
        switch_url: str = DEFAULT_SWITCH_URL,
    ):
        self.sovits_url = sovits_url
        self.switch_url = switch_url

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
        emotion: Optional[str],
        voice_id: str,
        voice_spec: Dict,
        output_path: str,
    ) -> bool:
        endpoint = _sovits_tts_endpoint(self.sovits_url)
        payload = {
            "text": text,
            "text_lang": _resolve_text_lang(text),
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


class CosyVoiceEngine(BaseTTSEngine):
    name = "cosyvoice"

    def __init__(
        self,
        default_spk_id: str = "",
        model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
    ):
        if AutoModel is None or torchaudio is None:
            raise RuntimeError(
                "CosyVoice 依赖缺失，请安装 cosyvoice 与 torchaudio 后再使用 cosyvoice 引擎"
            )
        self.default_spk_id = (default_spk_id or "").strip()
        self.model_dir = model_dir
        self.cosyvoice = AutoModel(model_dir=self.model_dir)

    def _resolve_spk_id(self, voice_id: str, voice_spec: Dict) -> str:
        preferred = (voice_spec.get("cosyvoice_spk_id") or "").strip()
        if preferred:
            return preferred
        if self.default_spk_id:
            return self.default_spk_id
        return (voice_id or "").strip()

    def prepare_voice(self, voice_id: str, voice_spec: Dict) -> bool:
        return True

    def _generate_line_sync(
        self,
        text: str,
        emotion: Optional[str],
        voice_id: str,
        voice_spec: Dict,
        output_path: str,
    ) -> bool:
        try:
            for out in self.cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text="",
                prompt_wav="",
                zero_shot_spk_id=voice_id,
                speed=1.1,
            ):
                torchaudio.save(
                    output_path, out["tts_speech"], self.cosyvoice.sample_rate
                )
                return True
        except Exception as exc:
            print(f"   CosyVoice 推理异常: {exc}")
            return False
        return False

    async def generate_line(
        self,
        text: str,
        emotion: Optional[str],
        voice_id: str,
        voice_spec: Dict,
        output_path: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._generate_line_sync,
            text,
            emotion,
            voice_id,
            voice_spec,
            output_path,
        )


def create_tts_engine(
    engine_name: str = DEFAULT_TTS_ENGINE,
    sovits_url: str = DEFAULT_SOVITS_URL,
    sovits_switch_url: str = DEFAULT_SWITCH_URL,
    cosyvoice_url: str = DEFAULT_COSYVOICE_URL,
    cosyvoice_mode: str = DEFAULT_COSYVOICE_MODE,
    cosyvoice_sample_rate: int = DEFAULT_COSYVOICE_SAMPLE_RATE,
    cosyvoice_spk_id: str = DEFAULT_COSYVOICE_SPK_ID,
) -> BaseTTSEngine:
    normalized = (engine_name or "").strip().lower()
    if normalized in {"gpt-sovits", "gpt_sovits", "sovits"}:
        return GptSoVitsEngine(sovits_url=sovits_url, switch_url=sovits_switch_url)
    if normalized in {"cosyvoice", "cosy"}:
        return CosyVoiceEngine(
            default_spk_id=cosyvoice_spk_id,
        )
    raise ValueError(
        f"不支持的 TTS 引擎: {engine_name}，"
        f"可选值: {', '.join(SUPPORTED_TTS_ENGINES)}"
    )


# 兼容旧调用路径：保留原有函数签名
async def tts_generate(
    text,
    voice_spec,
    output_path,
    sovits_url=DEFAULT_SOVITS_URL,
):
    engine = GptSoVitsEngine(sovits_url=sovits_url)
    return await engine.generate_line(
        text=text,
        emotion=None,
        voice_id=voice_spec.get("voice_id", DEFAULT_VOICE_ID),
        voice_spec=voice_spec,
        output_path=output_path,
    )
