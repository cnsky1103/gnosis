from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class CharacterProfile(BaseModel):
    name: str
    gender: Literal["male", "female", "unknown"]
    voice_archetype: str
    # GPT-SoVITS 专属：一旦分配，终身锁定
    ref_audio_path: Optional[str] = None
    ref_audio_text: Optional[str] = None
    description: Optional[str] = None


class ScriptLine(BaseModel):
    text: str
    speaker: str
    emotion: str = "neutral"
    type: Literal["dialogue", "narration", "thought"]


class CharacterExtraction(BaseModel):
    new_characters: List[CharacterProfile]


class ScriptResult(BaseModel):
    script: List[ScriptLine]
