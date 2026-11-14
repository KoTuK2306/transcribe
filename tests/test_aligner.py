"""Тесты transcribe.aligner: привязка слов к спикерам и склейка реплик.

Все данные синтетические: слова — dict {start, end, text}, интервалы
диаризации — dict {start, end, speaker}, как в кэше этапов.
"""

from transcribe.aligner import assign_speakers, group_into_lines


def make_word(start: float, end: float, text: str = "слово") -> dict:
    return {"start": start, "end": end, "text": text}


def make_turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


# ---------- assign_speakers ----------


def test_word_fully_inside_single_segment_gets_its_speaker():
    words = [make_word(1.0, 1.5), make_word(10.0, 10.5)]
    turns = [make_turn(0.0, 5.0, "A"), make_turn(8.0, 12.0, "B")]

    assign_speakers(words, turns)

    assert [w["speaker"] for w in words] == ["A", "B"]


def test_word_on_boundary_goes_to_speaker_with_max_overlap():
    # Граница интервалов на 5.0; слово 4.6–5.4: пересечение с A — 0.4 с,
    # с B — 0.4 с... сдвигаем, чтобы пересечения различались.
    words = [make_word(4.0, 5.5)]
    turns = [make_turn(0.0, 5.0, "A"), make_turn(5.0, 9.0, "B")]

    assign_speakers(words, turns)

    # overlap с A = 1.0 (4.0–5.0), с B = 0.5 (5.0–5.5) → A.
    assert words[0]["speaker"] == "A"


def test_word_with_greater_overlap_to_second_segment_gets_second_speaker():
    words = [make_word(4.5, 5.9)]
    turns = [make_turn(0.0, 5.0, "A"), make_turn(5.0, 9.0, "B")]

    assign_speakers(words, turns)

    # overlap с A = 0.5 (4.5–5.0), с B = 0.9 (5.0–5.9) → B.
    assert words[0]["speaker"] == "B"


def test_word_outside_all_segments_is_unknown():
    words = [make_word(20.0, 20.5)]
    turns = [make_turn(0.0, 5.0, "A")]

    assign_speakers(words, turns)

    assert words[0]["speaker"] == "UNKNOWN"


def test_word_touching_boundary_without_overlap_is_unknown():
    # Слово заканчивается ровно на старте интервала: пересечение = 0,
    # код требует overlap > 0 → UNKNOWN.
    words = [make_word(4.0, 5.0)]
    turns = [make_turn(5.0, 9.0, "B")]

    assign_speakers(words, turns)

    assert words[0]["speaker"] == "UNKNOWN"


def test_empty_words_list_returns_elapsed_time():
    elapsed = assign_speakers([], [make_turn(0.0, 5.0, "A")])

    assert isinstance(elapsed, float)


def test_empty_diarization_marks_all_words_unknown():
    words = [make_word(1.0, 1.5), make_word(2.0, 2.5)]

    assign_speakers(words, [])

    assert [w["speaker"] for w in words] == ["UNKNOWN", "UNKNOWN"]


def test_turns_are_sorted_internally_input_order_does_not_matter():
    words = [make_word(1.0, 1.5)]
    turns = [make_turn(8.0, 12.0, "B"), make_turn(0.0, 5.0, "A")]

    assign_speakers(words, turns)

    assert words[0]["speaker"] == "A"


# ---------- group_into_lines ----------


def test_adjacent_words_of_one_speaker_glue_into_single_line():
    words = [
        {"start": 0.0, "end": 0.5, "text": "привет", "speaker": "A"},
        {"start": 0.6, "end": 1.2, "text": " мир", "speaker": "A"},
    ]

    lines = group_into_lines(words)

    assert len(lines) == 1
    assert lines[0]["speaker"] == "A"
    assert lines[0]["text"] == "привет мир"
    assert lines[0]["start"] == 0.0
    assert lines[0]["end"] == 1.2


def test_speaker_change_starts_new_line():
    words = [
        {"start": 0.0, "end": 0.5, "text": "привет", "speaker": "A"},
        {"start": 0.6, "end": 1.2, "text": "мир", "speaker": "B"},
        {"start": 1.3, "end": 1.9, "text": "снова А", "speaker": "A"},
    ]

    lines = group_into_lines(words)

    assert [line["speaker"] for line in lines] == ["A", "B", "A"]


def test_pause_longer_than_max_gap_starts_new_line():
    words = [
        {"start": 0.0, "end": 0.5, "text": "привет", "speaker": "A"},
        {"start": 3.0, "end": 3.5, "text": "мир", "speaker": "A"},
    ]

    lines = group_into_lines(words, max_gap=2.0)

    assert len(lines) == 2


def test_pause_equal_to_max_gap_does_not_split():
    # Проверка границы: разрыв ровно 2.0 с не разрывает реплику
    # (в коде строгое > max_gap).
    words = [
        {"start": 0.0, "end": 1.0, "text": "привет", "speaker": "A"},
        {"start": 3.0, "end": 4.0, "text": "мир", "speaker": "A"},
    ]

    lines = group_into_lines(words, max_gap=2.0)

    assert len(lines) == 1


def test_long_line_is_split_by_max_len():
    words = [
        {"start": float(i), "end": i + 0.5, "text": f"слово{i}", "speaker": "A"}
        for i in range(10)
    ]

    lines = group_into_lines(words, max_len=20)

    assert len(lines) > 1
    assert all(line["speaker"] == "A" for line in lines)
    # Порядок слов сохраняется: тексты не перемешиваются.
    joined = " ".join(line["text"] for line in lines)
    assert joined == " ".join(f"слово{i}" for i in range(10))


def test_empty_word_list_gives_no_lines():
    assert group_into_lines([]) == []
