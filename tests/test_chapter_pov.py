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
