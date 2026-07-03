import json

from gnosis.proofread_web import ScriptStore


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_project_view_uses_script_characters_instead_of_character_db(tmp_path):
    projects_root = tmp_path / "projects"
    project_root = projects_root / "demo"
    project_root.mkdir(parents=True)

    _write_json(
        project_root / "script.json",
        {
            "characters": [
                {"name": "剧本人物", "voice": "script_voice"},
                "not-a-character",
                {"name": "剧本人物", "voice": "duplicate_voice"},
                {"name": "第二人物", "voice": "script_voice_2"},
            ],
            "script": [{"speaker": "剧本人物", "text": "你好"}],
        },
    )
    _write_json(
        project_root / "character_db.json",
        [
            {"name": "旧角色库人物", "voice": "db_voice"},
        ],
    )

    view = ScriptStore(projects_root).get_project_view("demo")

    assert view["characters"] == [
        {"name": "剧本人物", "voice": "script_voice"},
        {"name": "第二人物", "voice": "script_voice_2"},
    ]
    assert view["key_map"] == [
        {"key": "0", "name": "narrator"},
        {"key": "1", "name": "剧本人物"},
        {"key": "2", "name": "第二人物"},
    ]
