from typing import Dict, List, Tuple

# LLM 仅可使用以下标签
ALLOWED_CHARACTER_TAGS: List[str] = [
    "男-元气",
    "男-正太",
    "男-沉稳",
    "男-温柔",
    "男-慵懒",
    "男-大叔",
    "男-老人",
    "女-萝莉",
    "女-傲娇",
    "女-优雅",
    "女-御姐",
    "女-三无",
    "女-JK",
    "未知",
]

# 声线池映射: ['性别', '风格'] -> ['seed_id', ...]
VOICE_SEEDS_BY_TAG_PARTS: Dict[Tuple[str, str], List[str]] = {
    ("男", "元气"): ["zuole"],
    ("男", "正太"): ["zuole"],
    ("男", "沉稳"): ["zuole"],
    ("男", "温柔"): ["zuole"],
    ("男", "慵懒"): ["zuole"],
    ("男", "大叔"): ["zuole"],
    ("男", "老人"): ["zuole"],
    ("女", "萝莉"): ["zuole"],
    ("女", "傲娇"): ["zuole"],
    ("女", "优雅"): ["zuole"],
    ("女", "御姐"): ["zuole"],
    ("女", "三无"): ["zuole"],
    ("女", "JK"): ["zuole"],
    ("未知", "未知"): ["zuole"],
}

GENDER_BY_TAG_PREFIX = {"男": "male", "女": "female", "未知": "unknown"}

DEFAULT_CHARACTER_TAG = "未知"


def normalize_character_tag(tag: str) -> str:
    if tag in ALLOWED_CHARACTER_TAGS:
        return tag
    return DEFAULT_CHARACTER_TAG


def split_character_tag(tag: str) -> Tuple[str, str]:
    normalized = normalize_character_tag(tag)
    if normalized == "未知":
        return ("未知", "未知")
    if "-" not in normalized:
        return ("未知", "未知")
    prefix, style = normalized.split("-", 1)
    return (prefix, style)


def get_gender_from_tag(tag: str) -> str:
    prefix, _ = split_character_tag(tag)
    return GENDER_BY_TAG_PREFIX.get(prefix, "unknown")


def get_voice_seeds_for_tag(tag: str) -> List[str]:
    tag_parts = split_character_tag(tag)
    seeds = VOICE_SEEDS_BY_TAG_PARTS.get(tag_parts, [])
    if seeds:
        return seeds
    return VOICE_SEEDS_BY_TAG_PARTS[("未知", "未知")]

