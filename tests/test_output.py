"""Тесты transcribe.output: таймкоды, нумерация спикеров, формат строки."""

import pytest

from transcribe.output import (
    build_speaker_rename,
    fmt_mmss,
    fmt_timestamp,
    write_transcript,
)


# ---------- fmt_timestamp / fmt_mmss ----------

@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (59, "00:00:59"),
        (60, "00:01:00"),
        (195, "00:03:15"),          # пример из README
        (3599, "00:59:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (195.9, "00:03:15"),        # секунды усекаются, без округления
    ],
)
def test_fmt_timestamp(seconds, expected):
    assert fmt_timestamp(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00"),
        (9, "0:09"),
        (65, "1:05"),
        (3599, "59:59"),
        (3600, "60:00"),
        (65.9, "1:05"),             # усечение, как в fmt_timestamp
    ],
)
def test_fmt_mmss(seconds, expected):
    assert fmt_mmss(seconds) == expected


# ---------- build_speaker_rename ----------

def test_speakers_are_numbered_by_descending_line_count():
    lines = (
        [{"speaker": "A"}] * 3
        + [{"speaker": "B"}] * 5
        + [{"speaker": "C"}] * 1
    )
    rename = build_speaker_rename(lines)

    assert rename == {"B": "SPEAKER_00", "A": "SPEAKER_01", "C": "SPEAKER_02"}


def test_unknown_is_not_renumbered():
    lines = [{"speaker": "UNKNOWN"}, {"speaker": "A"}, {"speaker": "A"}]
    rename = build_speaker_rename(lines)

    assert rename["UNKNOWN"] == "UNKNOWN"
    assert rename["A"] == "SPEAKER_00"


def test_ties_keep_first_appearance_order():
    # При равном числе реплик порядок — порядок первого появления
    # (sorted стабилен).
    lines = [{"speaker": "B"}, {"speaker": "A"}, {"speaker": "A"}, {"speaker": "B"}]
    rename = build_speaker_rename(lines)

    assert rename == {"B": "SPEAKER_00", "A": "SPEAKER_01"}


def test_single_speaker_is_speaker_00():
    rename = build_speaker_rename([{"speaker": "A"}, {"speaker": "A"}])

    assert rename == {"A": "SPEAKER_00"}


# ---------- write_transcript ----------

def test_write_transcript_matches_readme_example_byte_for_byte(tmp_path):
    # Эталон — пример фрагмента результата из README.md.
    lines = [
        {"start": 195.0, "end": 198.0, "speaker": "A",
         "text": "Добрый день, давайте начнём встречу."},
        {"start": 202.0, "end": 206.0, "speaker": "B",
         "text": "Здравствуйте, у меня пара вопросов по бюджету."},
    ]
    rename = {"A": "SPEAKER_00", "B": "SPEAKER_01"}
    out_path = tmp_path / "transcript.txt"

    write_transcript(str(out_path), lines, rename)

    expected = (
        "[00:03:15] SPEAKER_00: Добрый день, давайте начнём встречу.\n"
        "[00:03:22] SPEAKER_01: Здравствуйте, у меня пара вопросов по бюджету.\n"
    )
    # write_transcript пишет в текстовом режиме: на Windows концы строк —
    # CRLF, на Linux — LF. Нормализуем, чтобы эталон сравнивался одинаково.
    actual = out_path.read_bytes().replace(b"\r\n", b"\n")
    assert actual == expected.encode("utf-8")


def test_write_transcript_unknown_passes_through(tmp_path):
    lines = [{"start": 0.0, "end": 1.0, "speaker": "UNKNOWN", "text": "фон"}]
    out_path = tmp_path / "out.txt"

    write_transcript(str(out_path), lines, {"UNKNOWN": "UNKNOWN"})

    assert out_path.read_text(encoding="utf-8") == "[00:00:00] UNKNOWN: фон\n"


def test_write_transcript_pathlib_and_unicode_filename(tmp_path):
    lines = [{"start": 1.0, "end": 2.0, "speaker": "A", "text": "текст"}]
    out_path = tmp_path / "транскрипт встречи.txt"

    write_transcript(str(out_path), lines, {"A": "SPEAKER_00"})

    assert out_path.read_text(encoding="utf-8").startswith("[00:00:01] SPEAKER_00:")
