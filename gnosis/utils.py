import re
import os
import unicodedata

def clean_text(text):
    if not text:
        return ""
    
    # 定义保留范围的正则表达式：
    # \u4e00-\u9fa5: 常用汉字 (基本平面)
    # \u3040-\u309f: 日语平假名
    # \u30a0-\u30ff: 日语片假名
    # a-zA-Z: 英文字母
    # 0-9: 数字
    pattern = re.compile(r'[^\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fa5a-zA-Z0-9]')
    
    # 将不在白名单内的所有字符（包括空格、各种标点）替换为空字符串
    cleaned = re.sub(pattern, '', text)
    
    return cleaned

def remove_code_fences_regex(text):
    # 匹配开头和结尾的 ``` 行（包括可能带有的语言标识符，如 ```python）
    text = text.strip()
    text = re.sub(r"^```.*?\n", "", text)
    text = re.sub(r"\n```.*?$", "", text)
    return text.strip()


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

def _is_segment_file(path):
    if not os.path.isfile(path):
        return False
    file_name = os.path.basename(path)
    if file_name.startswith("silence_"):
        return False
    extension = os.path.splitext(file_name)[1].lower()
    return extension in SUPPORTED_AUDIO_EXTENSIONS


def collect_sorted_segments(audio_dir):
    paths = [
        os.path.join(audio_dir, name)
        for name in os.listdir(audio_dir)
        if _is_segment_file(os.path.join(audio_dir, name))
    ]
    paths.sort()
    return paths

def is_punctuation_only_text(text):
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

def parse_script_payload(script_payload):
    if isinstance(script_payload, dict) and isinstance(script_payload.get("script"), list) and isinstance(script_payload.get("characters"), list):
        return script_payload["characters"], script_payload["script"]
    raise ValueError("script.json 格式不正确，期望为 {'characters': [...], 'script': [...]} 或 [...]")