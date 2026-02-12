# novel_cast/models.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict


# 1. 角色档案 (存放在全局表中)
class CharacterProfile(BaseModel):
    name: str
    gender: Literal["male", "female", "unknown"]
    # 让 LLM 选一个最贴切的声音原型，而不是具体的 TTS ID
    voice_archetype: Literal[
        "young_energetic_male",
        "mature_calm_male",
        "old_wise_male",
        "villain_male",
        "young_sweet_female",
        "mature_elegant_female",
        "old_kind_female",
        "villain_female",
        "narrator_standard",
        "child_neutral",
    ]
    description: Optional[str] = Field(None, description="角色的简短描述，用于辅助记录")


# 2. 剧本行 (精简版，省 Token)
class ScriptLine(BaseModel):
    text: str
    speaker: str  # 必须对应 CharacterProfile.name
    emotion: str
    type: Literal["dialogue", "narration", "thought"]


# 3. LLM 的完整返回包
class ChapterAnalysis(BaseModel):
    new_characters: List[CharacterProfile] = Field(
        description="本章节首次出现的新角色列表"
    )
    script: List[ScriptLine] = Field(description="本章节的剧本内容")
