import os
from typing import Dict, List, Tuple

DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v3.1")

# LLM 仅可使用以下标签
ALLOWED_CHARACTER_TAGS: List[str] = [
    "男-元气",
    "男-正太",
    "男-温柔",
    "男-普通",
    "男-大叔",
    "女-萝莉",
    "女-傲娇",
    "女-温柔",
    "女-御姐",
    "女-元气",
    "女-三无",
    "女-普通",
    "未知",
]

# 声线池映射: ['性别', '风格'] -> ['seed_id', ...]
VOICE_SEEDS_BY_TAG_PARTS: Dict[Tuple[str, str], List[str]] = {
    ("男", "元气"): ["baitie", "cangtai"],
    ("男", "正太"): ["xuerong", "zhijian"],
    ("男", "温柔"): ["liuming", "xunshi"],
    ("男", "普通"): ["yinxian", "logos", "duanya", "zuole"],
    ("男", "大叔"): ["maenna", "heijiaoS"],
    ("女", "萝莉"): ["wenmi"],
    ("女", "傲娇"): ["tianhuo"],
    ("女", "少女"): ["anjielina", "haruka"],
    ("女", "温柔"): ["raidian", "perfumer"],
    ("女", "御姐"): ["shenxun"],
    ("女", "三无"): ["red"],
    ("女", "普通"): ["perfumer", "yela"],
    ("未知", "未知"): ["logos"],
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
