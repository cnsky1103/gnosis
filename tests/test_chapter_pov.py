import json

from gnosis.chapter_pov import (
    ChapterPovStore,
    build_chapter_segments,
    find_chapter_title_matches,
    load_chapter_pov_entries,
    validate_chapter_pov_entries,
)
from gnosis.models import CharacterExtraction, ChapterPovEntry


def test_character_extraction_accepts_old_response_without_chapters():
    payload = {
        "new_characters": [
            {
                "name": "浅村悠太",
                "gender": "male",
                "voice_archetype": "男-普通",
                "description": "第一人称主角",
            }
        ]
    }

    extraction = CharacterExtraction.model_validate(payload)

    assert extraction.new_characters[0].name == "浅村悠太"
    assert extraction.chapters == []


def test_character_extraction_accepts_chapter_pov_entries():
    payload = {
        "new_characters": [],
        "chapters": [
            {
                "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                "pov_speaker": "绫濑沙季",
                "evidence": "标题末尾出现绫濑沙季",
            }
        ],
    }

    extraction = CharacterExtraction.model_validate(payload)

    assert extraction.chapters == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
            evidence="标题末尾出现绫濑沙季",
        )
    ]


def test_chapter_pov_store_preserves_reviewed_entries(tmp_path):
    path = tmp_path / "chapter_pov.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": "第四卷 9月3日（星期四）绫濑沙季",
                    "pov_speaker": "人工修正后的绫濑沙季",
                    "evidence": "人工审核",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）绫濑沙季",
                pov_speaker="绫濑沙季",
                evidence="LLM 抽取",
            )
        ]
    )

    assert entries == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="人工修正后的绫濑沙季",
            evidence="人工审核",
        )
    ]
    assert load_chapter_pov_entries(str(path))[0] == entries


def test_chapter_pov_store_deduplicates_collapsed_whitespace(tmp_path):
    path = tmp_path / "chapter_pov.json"
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷   9月3日（星期四）  绫濑沙季",
                pov_speaker="绫濑沙季",
            ),
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四） 绫濑沙季",
                pov_speaker="错误的重复项",
            ),
        ]
    )

    assert len(entries) == 1
    assert entries[0].pov_speaker == "绫濑沙季"


def test_chapter_pov_store_does_not_overwrite_invalid_json(tmp_path):
    path = tmp_path / "chapter_pov.json"
    invalid_content = "{not json"
    path.write_text(invalid_content, encoding="utf-8")
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）浅村悠太",
                pov_speaker="浅村悠太",
            )
        ]
    )

    assert entries == []
    assert path.read_text(encoding="utf-8") == invalid_content
    assert any(
        warning.startswith("chapter_pov.json invalid JSON:")
        for warning in store.last_warnings
    )


def test_chapter_pov_store_does_not_overwrite_non_array_root(tmp_path):
    path = tmp_path / "chapter_pov.json"
    original_content = json.dumps({"chapter_title": "不是数组"}, ensure_ascii=False)
    path.write_text(original_content, encoding="utf-8")
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）浅村悠太",
                pov_speaker="浅村悠太",
            )
        ]
    )

    assert entries == []
    assert path.read_text(encoding="utf-8") == original_content
    assert any(
        warning.startswith("chapter_pov.json root must be an array:")
        for warning in store.last_warnings
    )


def test_chapter_pov_store_repairs_malformed_list_entries(tmp_path):
    path = tmp_path / "chapter_pov.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chapter_title": "第四卷 9月3日（星期四）浅村悠太",
                    "pov_speaker": "浅村悠太",
                },
                "不是对象",
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）绫濑沙季",
                pov_speaker="绫濑沙季",
            )
        ]
    )

    assert entries == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]
    assert "chapter_pov entry is not an object at index 1" in store.last_warnings
    assert load_chapter_pov_entries(str(path)) == (entries, [])


def test_chapter_pov_store_repairs_empty_required_fields(tmp_path):
    path = tmp_path / "chapter_pov.json"
    path.write_text(
        json.dumps(
            [
                {"chapter_title": "", "pov_speaker": "浅村悠太"},
                {"chapter_title": "第四卷 9月3日（星期四）绫濑沙季", "pov_speaker": ""},
                {
                    "chapter_title": "第四卷 9月3日（星期四）浅村悠太",
                    "pov_speaker": "浅村悠太",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = ChapterPovStore(str(path))

    entries = store.merge_extracted(
        [
            ChapterPovEntry(
                chapter_title="第四卷 9月3日（星期四）绫濑沙季",
                pov_speaker="绫濑沙季",
            )
        ]
    )

    assert entries == [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]
    assert "chapter_pov entry missing chapter_title at index 0" in store.last_warnings
    assert "chapter_pov entry missing pov_speaker at index 1" in store.last_warnings
    assert load_chapter_pov_entries(str(path)) == (entries, [])


def test_find_chapter_title_matches_returns_ordered_offsets():
    text = (
        "序章内容\n\n"
        "第四卷 9月3日（星期四）浅村悠太\n正文一\n\n"
        "第四卷 9月3日（星期四）绫濑沙季\n正文二"
    )
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert warnings == []
    assert [match.entry.pov_speaker for match in matches] == ["浅村悠太", "绫濑沙季"]
    assert text[matches[0].start :].startswith("第四卷 9月3日（星期四）浅村悠太")
    assert text[matches[1].start :].startswith("第四卷 9月3日（星期四）绫濑沙季")


def test_missing_chapter_title_warns_and_does_not_stop_segmentation():
    text = "第四卷 9月3日（星期四）浅村悠太\n正文一"
    entries = [
        ChapterPovEntry(
            chapter_title="不存在的章节",
            pov_speaker="绫濑沙季",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert len(matches) == 1
    assert "chapter title not found: 不存在的章节" in warnings


def test_out_of_order_chapter_title_warns_and_is_skipped():
    text = (
        "第四卷 9月3日（星期四）浅村悠太\n正文一\n\n"
        "第四卷 9月3日（星期四）绫濑沙季\n正文二"
    )
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        ),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert [match.entry.pov_speaker for match in matches] == ["绫濑沙季"]
    assert "chapter title out of order: 第四卷 9月3日（星期四）浅村悠太" in warnings


def test_duplicate_title_warns_when_duplicate_appears_before_selected_match():
    text = (
        "重复章节\n旧正文\n\n"
        "下一章\n正文一\n\n"
        "重复章节\n新正文"
    )
    entries = [
        ChapterPovEntry(chapter_title="下一章", pov_speaker="浅村悠太"),
        ChapterPovEntry(chapter_title="重复章节", pov_speaker="绫濑沙季"),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert [match.entry.chapter_title for match in matches] == ["下一章"]
    assert "chapter title matched more than once: 重复章节" in warnings
    assert "chapter title out of order: 重复章节" in warnings


def test_out_of_order_title_with_later_duplicate_warns_and_is_skipped():
    text = (
        "第一章\n旧正文\n\n"
        "第二章\n正文二\n\n"
        "第一章\n重复正文"
    )
    entries = [
        ChapterPovEntry(chapter_title="第二章", pov_speaker="绫濑沙季"),
        ChapterPovEntry(chapter_title="第一章", pov_speaker="浅村悠太"),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)

    assert [match.entry.chapter_title for match in matches] == ["第二章"]
    assert "chapter title matched more than once: 第一章" in warnings
    assert "chapter title out of order: 第一章" in warnings


def test_title_matching_and_segments_normalize_crlf_line_endings():
    text = "序章内容\r\n\r\n第一章 浅村悠太\r\n正文一\r\n\r\n第二章 绫濑沙季\r\n正文二"
    entries = [
        ChapterPovEntry(chapter_title="第一章 浅村悠太", pov_speaker="浅村悠太"),
        ChapterPovEntry(chapter_title="第二章 绫濑沙季", pov_speaker="绫濑沙季"),
    ]

    matches, warnings = find_chapter_title_matches(text, entries)
    segments = build_chapter_segments(text, matches)

    assert warnings == []
    assert [match.start for match in matches] == [6, 20]
    assert segments[0].text == "序章内容\n\n"
    assert segments[1].text == "第一章 浅村悠太\n正文一\n\n"
    assert segments[1].chapter_title == "第一章 浅村悠太"
    assert segments[2].text == "第二章 绫濑沙季\n正文二"
    assert segments[2].pov_speaker == "绫濑沙季"


def test_text_before_first_match_becomes_no_pov_segment():
    text = "序章内容\n\n第四卷 9月3日（星期四）浅村悠太\n正文一"
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="浅村悠太",
        )
    ]

    matches, warnings = find_chapter_title_matches(text, entries)
    segments = build_chapter_segments(text, matches)

    assert warnings == []
    assert segments[0].chapter_title is None
    assert segments[0].pov_speaker is None
    assert segments[0].text == "序章内容\n\n"
    assert segments[1].chapter_title == "第四卷 9月3日（星期四）浅村悠太"
    assert segments[1].pov_speaker == "浅村悠太"


def test_validate_chapter_pov_entries_warns_for_unknown_speaker():
    entries = [
        ChapterPovEntry(
            chapter_title="第四卷 9月3日（星期四）浅村悠太",
            pov_speaker="不存在的人",
        )
    ]

    warnings = validate_chapter_pov_entries(entries, {"浅村悠太"})

    assert warnings == [
        "chapter POV speaker not in character_db.json: 不存在的人 (第四卷 9月3日（星期四）浅村悠太)"
    ]
