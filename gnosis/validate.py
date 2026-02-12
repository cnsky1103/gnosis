from typing import List, Literal
from pydantic import BaseModel


# --- 定义强类型结构 ---
class ScriptItem(BaseModel):
    text: str
    speaker: str
    gender: Literal["male", "female", "unknown"]
    # 限制情感标签，方便映射到 TTS 模型
    # emotion: Literal[
    #    "neutral", "angry", "happy", "sad", "fear", "surprise", "whisper", "shouting"
    # ] = "neutral"
    emotion: str
    type: Literal["dialogue", "narration", "thought"]


class ScriptResult(BaseModel):
    script: List[ScriptItem]
