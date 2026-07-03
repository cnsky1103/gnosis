from gnosis.chapter_pov import ChapterSegment
from gnosis.chunking import ChunkingConfig, split_chapter_segments_into_chunks


def test_chapter_segment_chunking_never_combines_chapter_titles():
    config = ChunkingConfig(target_chars=40, min_chars=0, max_chars=80)
    segments = [
        ChapterSegment(
            text="第一章 浅村悠太\n\n第一章的正文内容。" * 3,
            chapter_title="第一章 浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterSegment(
            text="第二章 绫濑沙季\n\n第二章的正文内容。" * 3,
            chapter_title="第二章 绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]

    chunks = split_chapter_segments_into_chunks(segments, config)

    assert chunks
    assert all(
        not ("第一章 浅村悠太" in chunk.text and "第二章 绫濑沙季" in chunk.text)
        for chunk in chunks
    )


def test_chapter_segment_chunks_carry_pov_speaker_metadata():
    config = ChunkingConfig(target_chars=40, min_chars=0, max_chars=80)
    segments = [
        ChapterSegment(
            text="第一章 浅村悠太\n\n第一章的正文内容。" * 3,
            chapter_title="第一章 浅村悠太",
            pov_speaker="浅村悠太",
        ),
        ChapterSegment(
            text="第二章 绫濑沙季\n\n第二章的正文内容。" * 3,
            chapter_title="第二章 绫濑沙季",
            pov_speaker="绫濑沙季",
        ),
    ]

    chunks = split_chapter_segments_into_chunks(segments, config)

    first_chapter_chunks = [chunk for chunk in chunks if "第一章的正文内容" in chunk.text]
    second_chapter_chunks = [chunk for chunk in chunks if "第二章的正文内容" in chunk.text]

    assert first_chapter_chunks
    assert second_chapter_chunks
    assert {chunk.pov_speaker for chunk in first_chapter_chunks} == {"浅村悠太"}
    assert {chunk.pov_speaker for chunk in second_chapter_chunks} == {"绫濑沙季"}


def test_no_pov_segment_preserves_empty_chapter_metadata():
    config = ChunkingConfig(target_chars=40, min_chars=0, max_chars=80)
    segments = [ChapterSegment(text="序章文本。\n\n还没有匹配到章节 POV。")]

    chunks = split_chapter_segments_into_chunks(segments, config)

    assert len(chunks) == 1
    assert chunks[0].chapter_title is None
    assert chunks[0].pov_speaker is None
