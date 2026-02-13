# novel_cast/state_manager.py
import json
import os
from .models import CharacterProfile


class CharacterManager:
    def __init__(self, db_path="data/character_db.json"):
        self.db_path = db_path
        self.characters = {}  # Name -> CharacterProfile
        self.load_db()

    def load_db(self):
        """加载已有的角色库"""
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for char_data in data:
                    char = CharacterProfile(**char_data)
                    self.characters[char.name] = char
        else:
            # 初始化默认旁白
            1

    def save_db(self):
        """保存更新后的角色库"""
        data = [char.model_dump() for char in self.characters.values()]
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_known_names(self):
        """生成 Prompt 用的简报字符串"""
        # 格式示例: "- 温水和彦 (male, young_energetic_male)"
        lines = []
        for char in self.characters.values():
            lines.append(f"- {char.name} ({char.gender}, {char.voice_archetype})")
        return "\n".join(lines)

    def add_character(self, profile: CharacterProfile):
        """注册新角色"""
        if profile.name not in self.characters:
            print(f"🆕 发现新角色: {profile.name} [{profile.voice_archetype}]")
            self.characters[profile.name] = profile
            return True
        return False
