import ast
import json
from pathlib import Path

from gnosis import pipeline
from gnosis.chunking import ChunkingConfig
from gnosis.llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE
from gnosis.models import CharacterProfile
from gnosis.state_manager import CharacterManager


class FakeCharacterManager:
    characters = {}

    def get_known_names_and_gender(self):
        return "旁白 (unknown)"


def test_main_wires_chapter_pov_path_to_passes():
    tree = ast.parse(Path("main.py").read_text(encoding="utf-8"))

    assert any(
        _is_chapter_pov_path_assignment(node) for node in ast.walk(tree)
    )
    assert _call_has_chapter_pov_path_keyword(tree, "run_pass1")
    assert _call_has_chapter_pov_path_keyword(tree, "run_pass2")


def _is_chapter_pov_path_assignment(node):
    if not isinstance(node, ast.Assign):
        return False
    if not any(
        isinstance(target, ast.Name) and target.id == "chapter_pov_path"
        for target in node.targets
    ):
        return False
    return _is_chapter_pov_join(node.value)


def _is_chapter_pov_join(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "project_root"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "chapter_pov.json"
    )


def _call_has_chapter_pov_path_keyword(tree, call_name):
    return any(
        _is_named_call(node, call_name)
        and any(
            keyword.arg == "chapter_pov_path"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "chapter_pov_path"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )


def _is_named_call(node, call_name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == call_name
    )


def test_pass1_prompt_requests_chapter_pov_output():
    prompt = PASS1_PROMPT_TEMPLATE.format(
        known_characters_str="浅村悠太\n绫濑沙季",
        allowed_character_tags="男-普通\n女-普通",
        project_pass1_prompt="无",
    )

    assert '"chapters"' in prompt
    assert "chapter_title" in prompt
    assert "pov_speaker" in prompt
    assert "章节" in prompt
    assert '{"new_characters":' in prompt
    assert '],"chapters":' in prompt


def test_run_pass1_persists_extracted_chapter_pov(tmp_path, monkeypatch):
    character_db_path = tmp_path / "character_db.json"
    chapter_pov_path = tmp_path / "chapter_pov.json"
    manager = CharacterManager(db_path=str(character_db_path))

    def fake_get_raw_response(**_kwargs):
        return (
            json.dumps(
                {
                    "new_characters": [
                        {
                            "name": "绫濑沙季",
                            "gender": "female",
                            "voice_archetype": "女-普通",
                            "description": "第一人称视角角色",
                        }
                    ],
                    "chapters": [
                        {
                            "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                            "pov_speaker": "绫濑沙季",
                            "evidence": "标题末尾出现绫濑沙季",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            str(tmp_path / "pass1.json"),
            False,
        )

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    pipeline.run_pass1(
        "第四卷 9月3日（星期四）绫濑沙季\n\n我走进教室。",
        manager,
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        chapter_pov_path=str(chapter_pov_path),
    )

    payload = json.loads(chapter_pov_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
            "pov_speaker": "绫濑沙季",
            "evidence": "标题末尾出现绫濑沙季",
        }
    ]


def test_run_pass1_accumulates_chapter_pov_across_chunks_in_order(
    tmp_path, monkeypatch
):
    character_db_path = tmp_path / "character_db.json"
    chapter_pov_path = tmp_path / "chapter_pov.json"
    manager = CharacterManager(db_path=str(character_db_path))
    seen_chunk_indexes = []

    chapter_by_chunk = {
        1: {
            "chapter_title": "第一章 浅村悠太",
            "pov_speaker": "浅村悠太",
            "evidence": "第一段标题",
        },
        2: {
            "chapter_title": "第二章 绫濑沙季",
            "pov_speaker": "绫濑沙季",
            "evidence": "第二段标题",
        },
        3: {
            "chapter_title": "第三章 浅村悠太",
            "pov_speaker": "浅村悠太",
            "evidence": "第三段标题",
        },
    }

    def fake_get_raw_response(**kwargs):
        chunk_index = kwargs["chunk_index"]
        seen_chunk_indexes.append(chunk_index)
        return (
            json.dumps(
                {
                    "new_characters": [],
                    "chapters": [chapter_by_chunk[chunk_index]],
                },
                ensure_ascii=False,
            ),
            str(tmp_path / f"pass1-{chunk_index}.json"),
            False,
        )

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    pipeline.run_pass1(
        "第一章 浅村悠太\n\n第二章 绫濑沙季\n\n第三章 浅村悠太",
        manager,
        ChunkingConfig(target_chars=1, min_chars=1, max_chars=20),
        cache_dir=str(tmp_path / "cache"),
        chapter_pov_path=str(chapter_pov_path),
    )

    assert seen_chunk_indexes == [1, 2, 3]
    assert json.loads(chapter_pov_path.read_text(encoding="utf-8")) == [
        chapter_by_chunk[1],
        chapter_by_chunk[2],
        chapter_by_chunk[3],
    ]


def test_pass2_prompt_includes_current_chapter_pov_context():
    prompt = PASS2_PROMPT_TEMPLATE.format(
        available_characters_str="浅村悠太 (male)\n绫濑沙季 (female)",
        previous_chunk_context_str="上一段摘要",
        chunk_index=1,
        total_chunks=3,
        chapter_title="第四卷 9月3日（星期四）绫濑沙季",
        pov_speaker="绫濑沙季",
        project_pass2_prompt="无",
    )

    assert "当前章节信息" in prompt
    assert "第四卷 9月3日（星期四）绫濑沙季" in prompt
    assert "第一人称视角：绫濑沙季" in prompt
    assert (
        "当前小说片段中所有非对话旁白、第一人称心理活动、第一人称叙述句的 `speaker` "
        "必须使用该角色名"
    ) in prompt


def test_run_pass2_formats_old_style_chunks_with_missing_chapter_pov(monkeypatch):
    captured_prompts = []

    def fake_get_raw_response(*, messages, **_kwargs):
        captured_prompts.append(messages[0]["content"])
        return '{"script":[]}', "/tmp/pass2.json", False

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    result = pipeline.run_pass2(
        "我走进教室。\n「早上好。」",
        FakeCharacterManager(),
        pass2_workers=1,
    )

    assert result == {"characters": [], "script": []}
    assert captured_prompts
    assert "章节标题：未提供" in captured_prompts[0]
    assert "第一人称视角：未提供" in captured_prompts[0]


def test_run_pass2_uses_reviewed_chapter_pov_metadata_in_prompt(
    tmp_path, monkeypatch
):
    title = "第四卷 9月3日（星期四）绫濑沙季"
    speaker = "绫濑沙季"
    chapter_pov_path = tmp_path / "chapter_pov.json"
    chapter_pov_path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": title,
                    "pov_speaker": speaker,
                    "evidence": "人工审核",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = CharacterManager(db_path=str(tmp_path / "character_db.json"))
    manager.add_character(
        CharacterProfile(
            name=speaker,
            gender="female",
            voice_archetype="女-普通",
            description="第一人称视角角色",
        )
    )
    captured_prompts = []

    def fake_get_raw_response(*, messages, **_kwargs):
        captured_prompts.append(messages[0]["content"])
        return '{"script":[]}', str(tmp_path / "pass2.json"), False

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    result = pipeline.run_pass2(
        f"{title}\n\n我走进教室。\n\n「早上好。」",
        manager,
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        pass2_workers=1,
        chapter_pov_path=str(chapter_pov_path),
    )

    assert result == {
        "characters": [manager.characters[speaker].model_dump()],
        "script": [],
    }
    assert any(f"章节标题：{title}" in prompt for prompt in captured_prompts)
    assert any(f"第一人称视角：{speaker}" in prompt for prompt in captured_prompts)


def test_run_pass2_partial_chapter_pov_match_falls_back_to_missing_metadata(
    tmp_path, monkeypatch, capsys
):
    first_title = "第一章 浅村悠太"
    first_speaker = "浅村悠太"
    missing_title = "第二章 绫濑沙季"
    second_speaker = "绫濑沙季"
    chapter_pov_path = tmp_path / "chapter_pov.json"
    chapter_pov_path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": first_title,
                    "pov_speaker": first_speaker,
                    "evidence": "人工审核",
                },
                {
                    "chapter_title": missing_title,
                    "pov_speaker": second_speaker,
                    "evidence": "人工审核",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manager = CharacterManager(db_path=str(tmp_path / "character_db.json"))
    for name, gender, archetype in [
        (first_speaker, "male", "男-普通"),
        (second_speaker, "female", "女-普通"),
    ]:
        manager.add_character(
            CharacterProfile(
                name=name,
                gender=gender,
                voice_archetype=archetype,
                description="章节视角角色",
            )
        )
    captured_prompts = []

    def fake_get_raw_response(*, messages, **_kwargs):
        captured_prompts.append(messages[0]["content"])
        return '{"script":[]}', str(tmp_path / "pass2.json"), False

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    pipeline.run_pass2(
        f"{first_title}\n\n我走进教室。\n\n下一章正文没有可匹配标题。",
        manager,
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        pass2_workers=1,
        chapter_pov_path=str(chapter_pov_path),
    )

    output = capsys.readouterr().out
    assert f"⚠️ chapter title not found: {missing_title}" in output
    assert captured_prompts
    assert "章节标题：未提供" in captured_prompts[0]
    assert "第一人称视角：未提供" in captured_prompts[0]
    assert f"章节标题：{first_title}" not in captured_prompts[0]
    assert f"第一人称视角：{first_speaker}" not in captured_prompts[0]


def test_run_pass2_missing_chapter_pov_path_warns_and_uses_missing_metadata(
    tmp_path, monkeypatch, capsys
):
    missing_chapter_pov_path = tmp_path / "missing" / "chapter_pov.json"
    captured_prompts = []

    def fake_get_raw_response(*, messages, **_kwargs):
        captured_prompts.append(messages[0]["content"])
        return '{"script":[]}', str(tmp_path / "pass2.json"), False

    monkeypatch.setattr(pipeline, "_get_raw_response", fake_get_raw_response)

    result = pipeline.run_pass2(
        "我走进教室。\n\n「早上好。」",
        FakeCharacterManager(),
        ChunkingConfig(target_chars=1000, min_chars=1, max_chars=1200),
        cache_dir=str(tmp_path / "cache"),
        pass2_workers=1,
        chapter_pov_path=str(missing_chapter_pov_path),
    )

    output = capsys.readouterr().out
    assert result == {"characters": [], "script": []}
    assert f"⚠️ chapter_pov.json missing: {missing_chapter_pov_path}" in output
    assert captured_prompts
    assert "章节标题：未提供" in captured_prompts[0]
    assert "第一人称视角：未提供" in captured_prompts[0]
