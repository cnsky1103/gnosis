from gnosis.utils import parse_script_payload
import json


SUPPORTED_TTS_ENGINES = ("cosyvoice", "gpt-sovits")

class BaseTTSEngine:
    name = "base"

    def init(self):
        pass

    async def generate_line(
        self,
        text: str,
        output_path: str,
        **kargs
    ) -> bool:
        raise NotImplementedError

    async def generate_script_tts(self, script_path, audio_dir, char = None, **kargs):
        raise NotImplementedError

    def group_jobs(self):
        # 按声线分组，减少模型切换次数
        self.grouped_jobs = {}
        for i, line in self.tts_jobs:
            speaker = line["speaker"]
            voice_id = self.speaker_to_voice.get(speaker, self.default_voice_id)
            self.grouped_jobs.setdefault(voice_id, []).append((i, line))
        for voice_id, jobs in self.grouped_jobs.items():
            print(f"   声线分组: {voice_id}, 句子数={len(jobs)}")



    def parse_script(self, script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            script_payload = json.load(f)
            return parse_script_payload(script_payload)



class TTSEngineProxy:
    _type: str
    _engine: BaseTTSEngine

    def __init__(self, type:str, engine: BaseTTSEngine):
        self._type = type
        self._engine = engine

    def init(self):
        pass

    async def generate_script_tts(self, script_path, audio_dir, char = None, **kargs):
        if self._type == 'cosyvoice':
            return await self._engine.generate_script_tts(script_path, audio_dir, char=char)

    def generate_line(self, text, output_path, **kargs):
        if self._type == 'cosyvoice':
            return self._engine.generate_line(text, output_path)
        else:
            pass
