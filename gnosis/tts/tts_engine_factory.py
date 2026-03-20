from gnosis.tts.tts_engine import TTSEngineProxy

from .sovits_engine import GptSoVitsEngine
from .cosy_voice_engine import CosyVoiceEngine

def create_sovits_engine(base_url: str):
    engine = GptSoVitsEngine(base_url)
    return TTSEngineProxy('sovits', engine)

def create_cosyvoice_engine(tts_workers):
    engine = CosyVoiceEngine(workers=tts_workers)
    return TTSEngineProxy('cosyvoice', engine)