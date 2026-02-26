from pydantic import BaseModel
from typing import List, Literal, Optional


class CharacterProfile(BaseModel):
    name: str
    gender: Literal["male", "female", "unknown"]
    voice_archetype: str = "未知"
    # 绑定后的声线种子ID，例如 "zuole"
    voice: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_audio_text: Optional[str] = None
    description: Optional[str] = None


class ScriptLine(BaseModel):
    text: str
    speaker: str
    emotion: str = "neutral"


class CharacterExtraction(BaseModel):
    new_characters: List[CharacterProfile]


class ScriptResult(BaseModel):
    script: List[ScriptLine]
