"""Тесты transcribe.cache: ключи, round-trip JSON, --no-cache, битый кэш."""

import json
import os

from transcribe.cache import StageCache, cache_key


# ---------- cache_key ----------

def test_cache_key_is_16_hex_chars():
    key = cache_key("meeting.mp3", "whisper", "large-v3")

    assert len(key) == 16
    assert all(ch in "0123456789abcdef" for ch in key)


def test_cache_key_is_deterministic():
    assert cache_key("a.mp3", "x", 1) == cache_key("a.mp3", "x", 1)


def test_cache_key_changes_with_any_part():
    base = cache_key("a.mp3", "whisper")
    assert cache_key("a.mp3", "diarization") != base
    assert cache_key("a.mp3", "whisper", "ru") != base
    assert cache_key("b.mp3", "whisper") != base


# ---------- StageCache round-trip ----------

def test_save_load_round_trip(tmp_path):
    cache = StageCache(str(tmp_path), "abc123")
    data = {"duration": 61.5, "words": [{"start": 0.0, "end": 0.5, "text": "привет"}]}

    cache.save("whisper", data)

    assert cache.load("whisper") == data
    assert (tmp_path / "abc123.whisper.json").is_file()


def test_save_creates_cache_dir(tmp_path):
    cache_dir = tmp_path / "подкаталог" / "кэш с пробелом"
    cache = StageCache(str(cache_dir), "key1")

    cache.save("diarization", [{"start": 0.0, "end": 1.0, "speaker": "A"}])

    assert (cache_dir / "key1.diarization.json").is_file()
    loaded = cache.load("diarization")
    assert loaded == [{"start": 0.0, "end": 1.0, "speaker": "A"}]


def test_load_returns_none_when_no_cache_file(tmp_path):
    cache = StageCache(str(tmp_path), "missing")

    assert cache.load("whisper") is None


def test_unicode_data_survives_round_trip(tmp_path):
    cache = StageCache(str(tmp_path), "ключ")
    data = [{"text": "привет, мир — «ёлки»"}]

    cache.save("whisper", data)

    # ensure_ascii=False: файл читается человеком, русский не экранирован.
    raw = (tmp_path / "ключ.whisper.json").read_text(encoding="utf-8")
    assert "привет" in raw
    assert cache.load("whisper") == data


# ---------- --no-cache (cache_dir=None) ----------

def test_no_cache_load_returns_none(tmp_path):
    cache = StageCache(None, "key")

    assert cache.load("whisper") is None


def test_no_cache_save_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = StageCache(None, "key")

    cache.save("whisper", {"words": []})

    assert list(tmp_path.iterdir()) == []


# ---------- битый кэш ----------

def test_broken_json_is_treated_as_missing_cache(tmp_path):
    cache = StageCache(str(tmp_path), "broken")
    (tmp_path / "broken.whisper.json").write_text('{"duration": 12.0, ', encoding="utf-8")

    # Код молча трактует битый JSON как отсутствие кэша (пересчёт этапа).
    assert cache.load("whisper") is None


def test_invalid_utf8_is_treated_as_missing_cache(tmp_path):
    cache = StageCache(str(tmp_path), "broken")
    (tmp_path / "broken.whisper.json").write_bytes(b"\xff\xfe\x00")

    assert cache.load("whisper") is None


def test_valid_json_with_wrong_shape_loads_as_is(tmp_path):
    # Поведение по коду: формат не валидируется, читается что записано.
    cache = StageCache(str(tmp_path), "old")
    (tmp_path / "old.whisper.json").write_text(
        json.dumps([{"start": 0.0, "end": 0.5, "text": "x"}]), encoding="utf-8"
    )

    loaded = cache.load("whisper")
    assert loaded == [{"start": 0.0, "end": 0.5, "text": "x"}]


# ---------- пути ----------

def test_os_path_accepts_pathlib_objects(tmp_path):
    cache = StageCache(os.fspath(tmp_path), "p1")
    cache.save("whisper", {"words": []})

    same = StageCache(os.fspath(tmp_path), "p1")
    assert same.load("whisper") == {"words": []}
