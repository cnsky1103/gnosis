import os
import requests

from gnosis.tts.tts_utils import DEFAULT_AUDIO_SAMPLE_RATE, PUNCTUATION_ONLY_SILENCE_MS, filter_script_jobs_by_character, write_silence_wav
from gnosis.utils import is_punctuation_only_text
# try:
#     from cosyvoice.cli.cosyvoice import AutoModel
# except ModuleNotFoundError:
#     AutoModel = None  # type: ignore[assignment]

# try:
#     import torchaudio
# except ModuleNotFoundError:
#     torchaudio = None  # type: ignore[assignment]

import asyncio
from typing import Dict

from gnosis.tts.tts_engine import BaseTTSEngine

DEFAULT_COSYVOICE_SPK_ID = 'logos'
DEFAULT_COSYVOICE_MODEL_DIR = (
    "/Users/sky/code/CosyVoice/pretrained_models/CosyVoice2-0.5B"
)

SERVER = "http://localhost:8090/generate"

class CosyVoiceEngine(BaseTTSEngine):
    name = "cosyvoice"

    def __init__(
        self,
        workers = 8
    ):
        # if AutoModel is None or torchaudio is None:
        #     raise RuntimeError(
        #         f"依赖缺失，cosyvoice={AutoModel} torchaudio={torchaudio}" 
        #     )
        self.model_dir = DEFAULT_COSYVOICE_MODEL_DIR 
        # self.cosyvoice = AutoModel(model_dir=self.model_dir)
        self.default_voice_id = "logos"
        self.workers = workers

    def _resolve_spk_id(self, voice_id: str, voice_spec: Dict) -> str:
        preferred = (voice_spec.get("cosyvoice_spk_id") or "").strip()
        if preferred:
            return preferred
        if self.default_voice_id:
            return self.default_voice_id
        return (voice_id or "").strip()

    def _generate_line_sync(
        self,
        text: str,
        voice_id: str,
        output_path: str,
    ) -> bool:
        response = requests.post(url=SERVER, json={
            "text": text,
            "voice_id": voice_id
        }, proxies={"http": None, "https": None})
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
                return True
        else:
            print(f"请求失败，状态码{response.status_code}")
            return False
        # try:
            # for out in self.cosyvoice.inference_zero_shot(
            #     tts_text=text,
            #     prompt_text="",
            #     prompt_wav="",
            #     zero_shot_spk_id=voice_id,
            #     speed=1.1,
            # ):
            #     torchaudio.save(
            #         output_path, out["tts_speech"], self.cosyvoice.sample_rate
            #     )
            #     return True
        # except Exception as exc:
        #     print(f"   CosyVoice 推理异常: {exc}")
        #     return False
        # return False

    async def generate_line(
        self,
        text: str,
        output_path: str,
        **kargs
    ) -> bool:
        #emotion = kargs.get("emotion")
        #voice_spec = kargs.get("voice_spec")
        voice_id = kargs.get("voice_id")
        return await asyncio.to_thread(
            self._generate_line_sync,
            text,
            voice_id,
            output_path,
        )

    async def generate_script_tts(self, script_path, audio_dir, char = None, **kargs):
        # 解析剧本，得到全部角色和剧本台词
        all_characters, script_list = self.parse_script(script_path)

        # 按角色过滤，得到实际说话的角色
        self.tts_jobs, _speakers = filter_script_jobs_by_character(script_list, char)

        if not self.tts_jobs:
            print('未命中任何角色，结束TTS')
            return

        self.speaker_to_voice = {}
        for char in all_characters:
            name = char.get("name")
            voice_id = char.get("voice") or self.default_voice_id
            self.speaker_to_voice[name] = voice_id

        self.group_jobs()

        pending_jobs = []
        for voice_id, jobs in self.grouped_jobs.items():
            for i, line in jobs:
                file_path = os.path.join(audio_dir, f"{i:04d}.wav")
                if os.path.exists(file_path):
                    continue
                pending_jobs.append((i, line, voice_id))

        if not pending_jobs:
            print("   所有音频片段已存在，跳过生成")
            return

        pending_jobs.sort(key=lambda item: item[0])
        job_queue = asyncio.Queue()
        for job in pending_jobs:
            job_queue.put_nowait(job)

        max_workers = max(1, min(self.workers, len(pending_jobs)))
        print(f"   CosyVoice 并发线程: {max_workers}, 待生成={len(pending_jobs)}")
        progress_lock = asyncio.Lock()
        state = {"done": 0, "error": None}

        async def worker():
            while True:
                if state["error"] is not None:
                    return
                try:
                    i, line, voice_id = job_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                file_path = os.path.join(audio_dir, f"{i:04d}.wav")
                try:
                    text = line.get("text", "")
                    emotion = line.get("emotion", "")
                    normalized_text = text if isinstance(text, str) else ""
                    if is_punctuation_only_text(normalized_text):
                        write_silence_wav(
                            file_path,
                            duration_ms=PUNCTUATION_ONLY_SILENCE_MS,
                            sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                        )
                        ok = True
                    else:
                        ok = await self.generate_line(
                            text=normalized_text,
                            output_path=file_path,
                            emotion=emotion if isinstance(emotion, str) else "",
                            voice_id=voice_id,
                        )
                    if not ok:
                        raise RuntimeError(
                            "TTS 调用失败:"
                            f" engine={self.name}, index={i},"
                            f" speaker={line['speaker']}, voice={voice_id}, text={text}"
                        )

                    async with progress_lock:
                        state["done"] += 1
                        print(
                            "   进度:"
                            f" {state['done']}/{len(pending_jobs)}"
                            f" -> {line['speaker']} (line {i + 1}/{len(self.tts_jobs)})"
                        )
                except Exception as exc:
                    async with progress_lock:
                        if state["error"] is None:
                            state["error"] = exc
                finally:
                    job_queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(max_workers)]
        await asyncio.gather(*tasks)
        if state["error"] is not None:
            raise state["error"]