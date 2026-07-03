from gnosis.llm_director import PASS1_PROMPT_TEMPLATE, PASS2_PROMPT_TEMPLATE


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
