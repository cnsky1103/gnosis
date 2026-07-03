import asyncio
import os
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from gnosis.tts.tts_engine import BaseTTSEngine
from gnosis.tts.tts_utils import DEFAULT_SOVITS_URL, is_han_char


HTTP = requests.Session()
HTTP.trust_env = False

DEFAULT_VOICE_ID = "logos"
DEFAULT_REF_DIR = "voice/ref"
GPT_WEIGHTS_DIR = "GPT_weights"
SOVITS_WEIGHTS_DIR = "SoVITS_weights"


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or DEFAULT_SOVITS_URL).strip().rstrip("/")
    if not normalized:
        return DEFAULT_SOVITS_URL
    return normalized


def _tts_endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效 GPT-SoVITS URL: {base_url}")
    return urlunsplit((parsed.scheme, parsed.netloc, "/tts", "", ""))


def _normalize_model_path(raw_model_path: Optional[str], model_dir: str) -> Optional[str]:
    if not raw_model_path:
        return None
    normalized = raw_model_path.strip().replace("\\", "/")
    if not normalized:
        return None
    if os.path.isabs(normalized) or normalized.startswith(f"{model_dir}/"):
        return normalized
    return f"{model_dir}/{normalized.lstrip('./')}"


def _resolve_path(base_dir: str, raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(base_dir, raw_path))


def _read_ref_lines(ref_file: str) -> Optional[List[str]]:
    if not os.path.exists(ref_file):
        return None

    with open(ref_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines or None


def _build_ref_file_path(ref_dir: str, voice_id: str) -> str:
    return os.path.join(ref_dir, f"{voice_id}.ref")


def _load_sovits_voice_spec(ref_file: str) -> Optional[Dict]:
    lines = _read_ref_lines(ref_file)
    if not lines:
        return None

    base_dir = os.path.dirname(os.path.abspath(ref_file))
    gpt_model_path = None
    sovits_model_path = None
    ref_audio_path = None
    prompt_lines = []

    # Standard .ref format:
    # 1 gpt_model_path, 2 sovits_model_path, 3 ref_audio_path, 4+ prompt_text.
    # Minimal format is also supported: 1 ref_audio_path, 2+ prompt_text.
    if len(lines) >= 4:
        gpt_model_path = _normalize_model_path(lines[0], GPT_WEIGHTS_DIR)
        sovits_model_path = _normalize_model_path(lines[1], SOVITS_WEIGHTS_DIR)
        ref_audio_path = _resolve_path(base_dir, lines[2])
        prompt_lines = lines[3:]
    elif len(lines) >= 2:
        ref_audio_path = _resolve_path(base_dir, lines[0])
        prompt_lines = lines[1:]
    else:
        ref_audio_path = _resolve_path(base_dir, lines[0])

    prompt_text = " ".join(prompt_lines).strip()
    if not ref_audio_path:
        return None

    return {
        "gpt_model_path": gpt_model_path,
        "sovits_model_path": sovits_model_path,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text or "这是系统默认旁白的参考文本。",
        "prompt_lang": "zh",
    }


def _build_character_voice_spec(char_info: Dict) -> Optional[Dict]:
    ref_audio_path = char_info.get("ref_audio_path")
    ref_audio_text = char_info.get("ref_audio_text")
    if not ref_audio_path or not ref_audio_text:
        return None
    return {
        "gpt_model_path": char_info.get("gpt_model_path"),
        "sovits_model_path": char_info.get("sovits_model_path"),
        "ref_audio_path": os.path.abspath(ref_audio_path),
        "prompt_text": ref_audio_text,
        "prompt_lang": "zh",
    }


class GptSoVitsEngine(BaseTTSEngine):
    name = "sovits"

    def __init__(
        self,
        base_url: str = DEFAULT_SOVITS_URL,
        ref_dir: str = DEFAULT_REF_DIR,
        default_voice_id: str = DEFAULT_VOICE_ID,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.sovits_url = _tts_endpoint(self.base_url)
        self.ref_dir = ref_dir
        self.default_voice_id = default_voice_id
        self.speaker_to_voice: Dict[str, str] = {}
        self.voice_specs: Dict[str, Dict] = {}
        self._current_gpt_model_path: Optional[str] = None
        self._current_sovits_model_path: Optional[str] = None
        self._lock = asyncio.Lock()

    def configure_voices(self, speaker_to_voice, characters):
        self.speaker_to_voice = dict(speaker_to_voice or {})
        self.voice_specs = self._build_voice_specs(characters or [])
        return self.voice_specs

    def _build_voice_specs(self, characters: List[Dict]) -> Dict[str, Dict]:
        voice_specs: Dict[str, Dict] = {}

        for char_info in characters:
            if not isinstance(char_info, dict):
                continue
            voice_id = (char_info.get("voice") or self.default_voice_id or "").strip()
            if not voice_id or voice_id in voice_specs:
                continue

            char_spec = _build_character_voice_spec(char_info)
            if char_spec:
                voice_specs[voice_id] = char_spec
                continue

            ref_file = _build_ref_file_path(self.ref_dir, voice_id)
            spec = _load_sovits_voice_spec(ref_file)
            if spec:
                voice_specs[voice_id] = spec

        default_ref_file = _build_ref_file_path(self.ref_dir, self.default_voice_id)
        default_spec = _load_sovits_voice_spec(default_ref_file)
        if default_spec and self.default_voice_id not in voice_specs:
            voice_specs[self.default_voice_id] = default_spec

        return voice_specs

    async def generate_line(
        self,
        text: str,
        output_path: str,
        **kwargs,
    ) -> bool:
        voice_id = (kwargs.get("voice_id") or self.default_voice_id).strip()
        voice_spec = self._resolve_voice_spec(voice_id)

        async with self._lock:
            return await asyncio.to_thread(
                self._generate_line_sync,
                text,
                output_path,
                voice_id,
                voice_spec,
            )

    def _resolve_voice_spec(self, voice_id: str) -> Dict:
        if voice_id in self.voice_specs:
            return self.voice_specs[voice_id]
        if self.default_voice_id in self.voice_specs:
            return self.voice_specs[self.default_voice_id]

        ref_file = _build_ref_file_path(self.ref_dir, voice_id)
        spec = _load_sovits_voice_spec(ref_file)
        if spec:
            self.voice_specs[voice_id] = spec
            return spec

        default_ref_file = _build_ref_file_path(self.ref_dir, self.default_voice_id)
        default_spec = _load_sovits_voice_spec(default_ref_file)
        if default_spec:
            self.voice_specs[self.default_voice_id] = default_spec
            return default_spec

        raise FileNotFoundError(
            "找不到 GPT-SoVITS 参考声线。需要至少提供"
            f" {_build_ref_file_path(self.ref_dir, self.default_voice_id)}"
        )

    def _generate_line_sync(
        self,
        text: str,
        output_path: str,
        voice_id: str,
        voice_spec: Dict,
    ) -> bool:
        if not self._prepare_voice(voice_spec):
            return False

        payload = {
            "text": text,
            "text_lang": self._resolve_text_lang(text),
            "ref_audio_path": os.path.abspath(voice_spec["ref_audio_path"]),
            "prompt_text": voice_spec.get("prompt_text", ""),
            "prompt_lang": voice_spec.get("prompt_lang", "zh"),
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": 0,
            "speed_factor": 1.1,
        }

        try:
            response = HTTP.post(self.sovits_url, json=payload, timeout=180)
        except requests.RequestException as exc:
            print(f"   GPT-SoVITS 请求异常: voice={voice_id}, error={exc}")
            return False

        if response.status_code != 200:
            print(
                "   GPT-SoVITS /tts 失败:"
                f" voice={voice_id}, status={response.status_code}, body={response.text[:200]}"
            )
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True

    def _prepare_voice(self, voice_spec: Dict) -> bool:
        gpt_model_path = voice_spec.get("gpt_model_path")
        sovits_model_path = voice_spec.get("sovits_model_path")

        if gpt_model_path and gpt_model_path != self._current_gpt_model_path:
            if not self._switch_model("/set_gpt_weights", gpt_model_path):
                return False
            self._current_gpt_model_path = gpt_model_path

        if sovits_model_path and sovits_model_path != self._current_sovits_model_path:
            if not self._switch_model("/set_sovits_weights", sovits_model_path):
                return False
            self._current_sovits_model_path = sovits_model_path

        return True

    def _switch_model(self, path: str, weights_path: str) -> bool:
        endpoint = f"{self.base_url}{path}"
        try:
            response = HTTP.get(endpoint, params={"weights_path": weights_path}, timeout=60)
        except requests.RequestException as exc:
            print(f"   GPT-SoVITS 模型切换异常: endpoint={path}, error={exc}")
            return False

        if response.status_code == 200:
            return True

        print(
            "   GPT-SoVITS 模型切换失败:"
            f" endpoint={path}, status={response.status_code}, body={response.text[:200]}"
        )
        return False

    @staticmethod
    def _resolve_text_lang(text: str) -> str:
        for ch in text:
            if ch.isalpha() and not is_han_char(ch):
                return "auto"
        return "zh"
