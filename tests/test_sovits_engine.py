import os

import pytest

from gnosis.tts import sovits_engine
from gnosis.tts.sovits_engine import GptSoVitsEngine


class FakeResponse:
    def __init__(self, status_code=200, content=b"RIFFfakeWAVE", text="ok"):
        self.status_code = status_code
        self.content = content
        self.text = text


class FakeHTTP:
    def __init__(self):
        self.get_calls = []
        self.post_calls = []

    def get(self, endpoint, params=None, timeout=None):
        self.get_calls.append((endpoint, params, timeout))
        return FakeResponse()

    def post(self, endpoint, json=None, timeout=None):
        self.post_calls.append((endpoint, json, timeout))
        return FakeResponse(content=b"RIFF\x00\x00\x00\x00WAVE")


def test_configure_voices_loads_ref_file(tmp_path):
    ref_dir = tmp_path / "voice" / "ref"
    ref_dir.mkdir(parents=True)
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (ref_dir / "logos.ref").write_text(
        "\n".join(
            [
                "/models/logos-gpt.ckpt",
                "/models/logos-sovits.pth",
                str(ref_audio),
                "参考文本。",
            ]
        ),
        encoding="utf-8",
    )

    engine = GptSoVitsEngine(ref_dir=str(ref_dir))
    specs = engine.configure_voices({"旁白": "logos"}, [{"name": "旁白", "voice": "logos"}])

    assert specs["logos"]["gpt_model_path"] == "/models/logos-gpt.ckpt"
    assert specs["logos"]["sovits_model_path"] == "/models/logos-sovits.pth"
    assert specs["logos"]["ref_audio_path"] == str(ref_audio)
    assert specs["logos"]["prompt_text"] == "参考文本。"


@pytest.mark.asyncio
async def test_generate_line_switches_weights_and_posts_tts(tmp_path, monkeypatch):
    fake_http = FakeHTTP()
    monkeypatch.setattr(sovits_engine, "HTTP", fake_http)

    output_path = tmp_path / "0000.wav"
    engine = GptSoVitsEngine(base_url="http://127.0.0.1:9880", ref_dir=str(tmp_path))
    engine.voice_specs = {
        "logos": {
            "gpt_model_path": "/models/logos-gpt.ckpt",
            "sovits_model_path": "/models/logos-sovits.pth",
            "ref_audio_path": str(tmp_path / "ref.wav"),
            "prompt_text": "参考文本。",
            "prompt_lang": "zh",
        }
    }

    ok = await engine.generate_line("你好，世界", str(output_path), voice_id="logos")

    assert ok is True
    assert output_path.exists()
    assert [call[0] for call in fake_http.get_calls] == [
        "http://127.0.0.1:9880/set_gpt_weights",
        "http://127.0.0.1:9880/set_sovits_weights",
    ]
    assert fake_http.post_calls[0][0] == "http://127.0.0.1:9880/tts"
    payload = fake_http.post_calls[0][1]
    assert payload["text"] == "你好，世界"
    assert payload["text_lang"] == "zh"
    assert payload["prompt_lang"] == "zh"
    assert payload["media_type"] == "wav"


@pytest.mark.asyncio
async def test_generate_line_uses_auto_lang_for_non_han_text(tmp_path, monkeypatch):
    fake_http = FakeHTTP()
    monkeypatch.setattr(sovits_engine, "HTTP", fake_http)

    output_path = tmp_path / "0000.wav"
    engine = GptSoVitsEngine(base_url="http://127.0.0.1:9880", ref_dir=str(tmp_path))
    engine.voice_specs = {
        "logos": {
            "ref_audio_path": os.path.join(str(tmp_path), "ref.wav"),
            "prompt_text": "参考文本。",
            "prompt_lang": "zh",
        }
    }

    ok = await engine.generate_line("Hello 世界", str(output_path), voice_id="logos")

    assert ok is True
    assert fake_http.post_calls[0][1]["text_lang"] == "auto"
