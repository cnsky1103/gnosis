import requests, os

SOVITS_URL = "http://127.0.0.1:9880"


async def tts_generate(text, speaker, char_manager, output_path):
    # 旁白走默认声线，角色走锁定声线
    if speaker == "narrator":
        ref_path = "./seeds/narrator.wav"
        ref_text = "这是系统默认旁白的参考文本。"
    else:
        char = char_manager.characters.get(speaker)
        ref_path = char.ref_audio_path
        ref_text = char.ref_audio_text

    params = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": os.path.abspath(ref_path),
        "prompt_text": ref_text,
        "prompt_lang": "zh",
    }

    r = requests.get(SOVITS_URL, params=params)
    if r.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(r.content)
        return True
    return False
