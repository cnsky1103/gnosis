from gnosis.tts.tts_utils import filter_script_jobs_by_character


def test_filter_script_jobs_by_character_limits_matched_character_only():
    script_lines = []
    for index in range(40):
        script_lines.append({"speaker": "目标", "text": f"目标 {index}"})
        script_lines.append({"speaker": "其他", "text": f"其他 {index}"})

    jobs, speakers = filter_script_jobs_by_character(
        script_lines,
        "目标",
        limit=30,
    )

    assert len(jobs) == 30
    assert speakers == {"目标", "其他"}
    assert jobs[0] == (0, {"speaker": "目标", "text": "目标 0"})
    assert jobs[-1] == (58, {"speaker": "目标", "text": "目标 29"})


def test_filter_script_jobs_by_character_limit_does_not_require_character():
    script_lines = [
        {"speaker": "甲", "text": "一"},
        {"speaker": "乙", "text": "二"},
        {"speaker": "甲", "text": "三"},
    ]

    jobs, speakers = filter_script_jobs_by_character(script_lines, "", limit=2)

    assert jobs == [
        (0, {"speaker": "甲", "text": "一"}),
        (1, {"speaker": "乙", "text": "二"}),
    ]
    assert speakers == {"甲", "乙"}
