from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# --- Pass 1: 角色提取模型 ---
class CharacterProfile(BaseModel):
    name: str
    gender: Literal["male", "female", "unknown"]
    voice_archetype: Literal[
        "young_energetic_male",
        "mature_calm_male",
        "old_wise_male",
        "villain_male",
        "young_sweet_female",
        "mature_elegant_female",
        "old_kind_female",
        "villain_female",
        "child_neutral",
    ]
    description: Optional[str] = Field(None, description="角色的简短描述")


class CharacterExtraction(BaseModel):
    new_characters: List[CharacterProfile] = Field(default_factory=list)


# --- Pass 2: 剧本生成模型 ---
class ScriptLine(BaseModel):
    text: str
    speaker: str  # 可以是具体的角色名，也可以是 "narrator"
    emotion: Literal[
        "neutral", "angry", "happy", "sad", "fear", "surprise", "whisper", "shouting"
    ] = "neutral"
    type: Literal["dialogue", "narration", "thought"]


class ScriptResult(BaseModel):
    script: List[ScriptLine]
