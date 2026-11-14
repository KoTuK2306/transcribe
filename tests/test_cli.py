"""Тесты transcribe.cli: значения по умолчанию (таблица README) и --help.

cuda_available и hf_token_from_env подменяются, чтобы тесты не импортировали
torch и не зависели от окружения; defaults вычисляются в parse_args при каждом
вызове, поэтому подмена через monkeypatch работает.
"""

import pytest

import transcribe.cli as cli


def parse(monkeypatch, argv, *, cuda=False, token=None):
    monkeypatch.setattr(cli, "cuda_available", lambda: cuda)
    monkeypatch.setattr(cli, "hf_token_from_env", lambda: token)
    monkeypatch.setattr("sys.argv", ["transcribe.py", *argv])
    return cli.parse_args()


# ---------- значения по умолчанию (таблица README «Запуск») ----------

def test_defaults_match_readme(monkeypatch):
    args = parse(monkeypatch, ["meeting.mp3"], cuda=False)

    assert args.audio == "meeting.mp3"
    assert args.out == "transcript.txt"
    assert args.model == "large-v3"
    assert args.language is None            # автоопределение
    assert args.device == "cpu"             # CUDA «недоступна» (патч выше)
    assert args.compute_type == "float16"
    assert args.min_speakers is None        # авто
    assert args.max_speakers is None        # авто
    assert args.cache_dir == "cache"
    assert args.no_cache is False
    assert args.hf_token is None            # HF_TOKEN не задан (патч выше)


def test_device_defaults_to_cuda_when_available(monkeypatch):
    args = parse(monkeypatch, ["meeting.mp3"], cuda=True)

    assert args.device == "cuda"


def test_hf_token_default_comes_from_env(monkeypatch):
    args = parse(monkeypatch, ["meeting.mp3"], token="hf_test_token")

    assert args.hf_token == "hf_test_token"


def test_explicit_flags_override_defaults(monkeypatch):
    args = parse(monkeypatch, [
        "meeting.mp3",
        "--out", "result.txt",
        "--model", "medium",
        "--language", "ru",
        "--device", "cpu",
        "--compute-type", "int8",
        "--min-speakers", "2",
        "--max-speakers", "3",
        "--cache-dir", "my-cache",
        "--no-cache",
        "--hf-token", "hf_xxx",
    ])

    assert args.out == "result.txt"
    assert args.model == "medium"
    assert args.language == "ru"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.min_speakers == 2
    assert args.max_speakers == 3
    assert args.cache_dir == "my-cache"
    assert args.no_cache is True
    assert args.hf_token == "hf_xxx"


# ---------- --help ----------

def test_help_exits_zero_and_lists_options(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse(monkeypatch, ["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for option in ("--out", "--model", "--language", "--device",
                   "--compute-type", "--min-speakers", "--max-speakers",
                   "--cache-dir", "--no-cache", "--hf-token"):
        assert option in out
