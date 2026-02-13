ARCHETYPE_TO_EDGE_TTS = {
    "narrator_standard": "zh-CN-YunxiNeural",  # 旁白：云希（沉稳男）
    "young_energetic_male": "zh-CN-YunyangNeural",  # 少年：云扬（新闻/专业）
    "mature_calm_male": "zh-CN-YunxiNeural",  # 成男：云希
    "old_wise_male": "zh-CN-YunzeNeural",  # 老年：云泽（虽然是老模型，凑合用）
    "villain_male": "zh-CN-YunyangNeural",  # 反派：可以通过调整 pitch/rate 实现
    "young_sweet_female": "zh-CN-XiaoxiaoNeural",  # 少女：晓晓（活泼）
    "mature_elegant_female": "zh-CN-XiaoyiNeural",  # 御姐：晓伊
    "old_kind_female": "zh-CN-LiaoNing-XiaobeiNeural",  # 这里的选择比较少，可能需要找特定方言或语气
    "child_neutral": "zh-CN-XiaoxiaoNeural",  # 儿童：通常用女声调高音调
}


def resolve_voice(character_name, char_manager):
    # 1. 从 manager 拿到角色的 profile
    profile = char_manager.characters.get(character_name)

    if not profile:
        # 如果 LLM 幻觉生成了没注册的名字，兜底
        return ARCHETYPE_TO_EDGE_TTS["narrator_standard"]

    # 2. 查表返回 ID
    return ARCHETYPE_TO_EDGE_TTS.get(profile.voice_archetype, "zh-CN-YunxiNeural")
