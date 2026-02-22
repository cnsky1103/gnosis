import os
import requests
from urllib.parse import urlsplit, urlunsplit

DEFAULT_SOVITS_URL = "http://127.0.0.1:9880/tts"
DEFAULT_VOICE_ID = "logos"
DEFAULT_SWITCH_URL = "http://127.0.0.1:9880"
GPT_WEIGHTS_DIR = "GPT_weights"
SOVITS_WEIGHTS_DIR = "SoVITS_weights"

HTTP = requests.Session()
HTTP.trust_env = False


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


def _tts_endpoint(url_or_endpoint):
    parsed = urlsplit(url_or_endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效URL: {url_or_endpoint}")
    if parsed.path.rstrip("/") == "/tts":
        return url_or_endpoint
    return urlunsplit((parsed.scheme, parsed.netloc, "/tts", "", ""))


def _load_voice_spec(ref_file):
    if not os.path.exists(ref_file):
        return None

    with open(ref_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return None

    base_dir = os.path.dirname(ref_file)
    gpt_model_path = None
    sovits_model_path = None
    ref_audio_path = None
    prompt_text = None

    # .ref 标准格式:
    # 1 gpt_model_path, 2 sovits_model_path, 3 ref_audio_path, 4+ prompt_text
    if len(lines) >= 4:
        gpt_model_path = _normalize_model_path(lines[0], GPT_WEIGHTS_DIR)
        sovits_model_path = _normalize_model_path(lines[1], SOVITS_WEIGHTS_DIR)
        ref_audio_path = _resolve_path(base_dir, lines[2])
        prompt_text = " ".join(lines[3:])
    # 兼容最简格式: 1 ref_audio_path, 2+ prompt_text
    elif len(lines) >= 2:
        ref_audio_path = _resolve_path(base_dir, lines[0])
        prompt_text = " ".join(lines[1:])
    else:
        ref_audio_path = _resolve_path(base_dir, lines[0])

    if not ref_audio_path:
        return None

    return {
        "gpt_model_path": gpt_model_path,
        "sovits_model_path": sovits_model_path,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text or "这是系统默认旁白的参考文本。",
    }


def _build_ref_file_path(seeds_dir, voice_id):
    return os.path.join(seeds_dir, f"{voice_id}.ref")


def build_voice_registry(characters, seeds_dir="voice/ref"):
    # 从 script.json 的 characters 构建:
    # 1) speaker -> voice_id
    # 2) voice_id -> voice_spec
    speaker_to_voice = {}
    voice_specs = {}

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
        spec = _load_voice_spec(ref_file)
        if not spec:
            continue
        voice_specs[voice_id] = spec

    default_ref_file = _build_ref_file_path(seeds_dir, DEFAULT_VOICE_ID)
    default_spec = _load_voice_spec(default_ref_file)
    if not default_spec:
        raise FileNotFoundError(f"默认声线ref不存在: {default_ref_file}")

    if DEFAULT_VOICE_ID not in voice_specs:
        voice_specs[DEFAULT_VOICE_ID] = default_spec

    return speaker_to_voice, voice_specs, DEFAULT_VOICE_ID


def switch_character(sovits_path, gpt_path, switch_url=DEFAULT_SWITCH_URL):
    # api_v2: /set_gpt_weights + /set_sovits_weights
    base = _base_url(switch_url)
    gpt_endpoint = f"{base}/set_gpt_weights"
    sovits_endpoint = f"{base}/set_sovits_weights"

    try:
        gpt_resp = HTTP.get(
            gpt_endpoint, params={"weights_path": gpt_path}, timeout=30
        )
        sovits_resp = HTTP.get(
            sovits_endpoint, params={"weights_path": sovits_path}, timeout=30
        )
    except requests.RequestException as e:
        print(f"   模型切换异常: {e}")
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


async def tts_generate(
    text,
    voice_spec,
    output_path,
    sovits_url=DEFAULT_SOVITS_URL,
):
    endpoint = _tts_endpoint(sovits_url)
    text_lang = _resolve_text_lang(text)
    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": os.path.abspath(voice_spec["ref_audio_path"]),
        "prompt_text": voice_spec["prompt_text"],
        "prompt_lang": "zh",
        "text_split_method": "cut5",
        "batch_size": 1,
        "streaming_mode": False,
        "speed_factor": 1.1
    }

    try:
        r = HTTP.post(endpoint, json=payload, timeout=120)
    except requests.RequestException:
        return False
    if r.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(r.content)
        return True
    print(f"   /tts 失败({r.status_code}): {r.text[:160]}")
    return False
