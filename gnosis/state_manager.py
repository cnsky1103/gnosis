import json, os, hashlib
from .models import CharacterProfile


class CharacterManager:
    def __init__(self, db_path="data/character_db.json", seeds_dir="./seeds"):
        self.db_path = db_path
        self.seeds_dir = seeds_dir
        self.characters = {}
        self.load_db()

    def load_db(self):
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

        # 1. 定位到具体的原型文件夹
        # 路径示例: ./seeds/female/young_sweet/
        archetype_dir = os.path.join(
            self.seeds_dir, profile.gender, profile.voice_archetype
        )

        # 2. 如果原型文件夹不存在，退而求其次走性别大类
        if not os.path.exists(archetype_dir):
            archetype_dir = os.path.join(self.seeds_dir, profile.gender)

        if os.path.exists(archetype_dir):
            seeds = sorted([f for f in os.listdir(archetype_dir) if f.endswith(".wav")])
            if seeds:
                # 在该特定原型分类下进行哈希，保证同一类型的角色分到不同的种子
                idx = int(hashlib.md5(profile.name.encode()).hexdigest(), 16) % len(
                    seeds
                )
                seed_name = seeds[idx]

                profile.ref_audio_path = os.path.join(archetype_dir, seed_name)
                # 加载参考文本
                txt_path = profile.ref_audio_path.rsplit(".", 1)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        profile.ref_audio_text = f.read().strip()

        self.characters[profile.name] = profile
        return True

    def get_known_names(self):
        return "\n".join(
            [
                f"- {c.name} ({c.gender}, {c.voice_archetype})"
                for c in self.characters.values()
            ]
        )
