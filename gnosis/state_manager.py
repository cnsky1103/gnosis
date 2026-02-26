import json, os, hashlib
from .models import CharacterProfile
from .config import (
    get_gender_from_tag,
    get_voice_seeds_for_tag,
    normalize_character_tag,
)


class CharacterManager:
    def __init__(self, db_path="data/character_db.json", seeds_dir="voice/ref/sovits"):
        self.db_path = db_path
        self.seeds_dir = seeds_dir
        self.characters = {}
        self.load_db()

    def load_db(self):
        self.characters = {}
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.characters[item["name"]] = CharacterProfile(**item)

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.model_dump() for c in self.characters.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add_character(self, profile: CharacterProfile):
        if profile.name in self.characters:
            return False

        # 标签标准化，并由标签反推出性别，避免 LLM 性别字段与标签冲突
        profile.voice_archetype = normalize_character_tag(profile.voice_archetype)
        profile.gender = get_gender_from_tag(profile.voice_archetype)

        self.characters[profile.name] = profile
        return True

    def _pick_seed_id(self, profile: CharacterProfile):
        seed_ids = get_voice_seeds_for_tag(profile.voice_archetype)
        if not seed_ids:
            return None
        idx = int(hashlib.md5(profile.name.encode()).hexdigest(), 16) % len(seed_ids)
        return seed_ids[idx]

    def assign_voice_to_character(self, profile: CharacterProfile, overwrite: bool = False):
        if profile.voice and not overwrite:
            return

        seed_id = self._pick_seed_id(profile)
        if not seed_id:
            return

        profile.voice = seed_id
        ref_file = os.path.join(self.seeds_dir, f"{seed_id}.ref")
        self._load_seed_ref(profile, ref_file)

    def assign_voices(self, overwrite: bool = False):
        for profile in self.characters.values():
            self.assign_voice_to_character(profile, overwrite=overwrite)

    def _load_seed_ref(self, profile: CharacterProfile, ref_file: str):
        if not os.path.exists(ref_file):
            return

        with open(ref_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                lines.append(stripped)

        if not lines:
            return

        def split_ref_kv(raw_line: str):
            for separator in ("=", ":"):
                if separator not in raw_line:
                    continue
                key, value = raw_line.split(separator, 1)
                key = key.strip().lower()
                value = value.strip()
                if key and value:
                    return key, value
            return None, None

        def looks_like_audio_path(raw_value: str):
            normalized = str(raw_value or "").strip().lower()
            return normalized.endswith(
                (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")
            )

        # key-value 格式（兼容 cosyvoice ref）
        kv_ref_audio_path = None
        kv_prompt_parts = []
        has_kv_line = False
        passthrough_lines = []
        for raw_line in lines:
            key, value = split_ref_kv(raw_line)
            if not key:
                passthrough_lines.append(raw_line)
                continue
            has_kv_line = True
            if key in {"ref_audio_path", "ref_audio", "ref_wav", "prompt_wav", "wav"}:
                kv_ref_audio_path = value
            elif key in {"prompt_text", "prompt", "text"}:
                kv_prompt_parts.append(value)

        if has_kv_line:
            if not kv_ref_audio_path and passthrough_lines:
                candidate = passthrough_lines[0]
                if looks_like_audio_path(candidate):
                    kv_ref_audio_path = candidate
                    passthrough_lines = passthrough_lines[1:]
            if kv_ref_audio_path:
                ref_wav_path = kv_ref_audio_path
                if not os.path.isabs(ref_wav_path):
                    ref_wav_path = os.path.abspath(
                        os.path.join(os.path.dirname(ref_file), ref_wav_path)
                    )
                profile.ref_audio_path = ref_wav_path
            if kv_prompt_parts:
                profile.ref_audio_text = " ".join(kv_prompt_parts)
            elif passthrough_lines:
                profile.ref_audio_text = " ".join(passthrough_lines)
            return

        # .ref 标准格式:
        # 1 ckpt, 2 pth, 3 ref wav path, 4+ 参考文本
        if len(lines) >= 4:
            ref_wav_path = lines[2]
            if not os.path.isabs(ref_wav_path):
                ref_wav_path = os.path.abspath(
                    os.path.join(os.path.dirname(ref_file), ref_wav_path)
                )
            profile.ref_audio_path = ref_wav_path
            profile.ref_audio_text = " ".join(lines[3:])
            return

        # 兼容最简格式: 第一行是 wav path，第二行是文本
        ref_wav_path = lines[0]
        if not os.path.isabs(ref_wav_path):
            ref_wav_path = os.path.abspath(
                os.path.join(os.path.dirname(ref_file), ref_wav_path)
            )
        profile.ref_audio_path = ref_wav_path
        if len(lines) >= 2:
            profile.ref_audio_text = " ".join(lines[1:])

    def get_known_names(self):
        return "\n".join(
            [
                f"- {c.name} ({c.gender}, {c.voice_archetype})"
                for c in self.characters.values()
            ]
        )

    def get_known_names_and_gender(self):
        return "\n".join([f"- {c.name} ({c.gender})" for c in self.characters.values()])
