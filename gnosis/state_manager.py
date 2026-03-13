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

    def assign_voice_to_character(self, profile: CharacterProfile):
        seed_id = self._pick_seed_id(profile)
        if not seed_id:
            return

        profile.voice = seed_id

    def assign_voices(self, overwrite: bool = False):
        for profile in self.characters.values():
            self.assign_voice_to_character(profile)


    def get_known_names(self):
        return "\n".join(
            [
                f"- {c.name} ({c.gender}, {c.voice_archetype})"
                for c in self.characters.values()
            ]
        )

    def get_known_names_and_gender(self):
        return "\n".join([f"- {c.name} ({c.gender})" for c in self.characters.values()])
