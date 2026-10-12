"""URL to SRT: prefer platform captions, automatically fall back to Whisper."""

import argparse
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from . import platforms
from .platforms import FetchError, report
from .transcription import DEFAULT_MODEL_DIR, load_model, transcribe_to_srt


def transcription_language(language):
    if language is None:
        return None
    # Platform dialect/script codes are language hints, not translation requests.
    code = platforms.language_key(language.replace("_", "-")).split("-")[0]
    code = {"iw": "he", "nb": "no", "nn": "no"}.get(code, code)
    from whisper.tokenizer import LANGUAGES
    if code not in LANGUAGES:
        raise FetchError(f"Whisper 不支持指定语言：{language}", "language")
    return code


def fetch_srt(url, *, language=None, model_path=DEFAULT_MODEL_DIR, chrome_profile=None):
    """Return validated SRT; no final subtitle files are persisted."""
    video = platforms.normalize_url(url)
    if language and not re.fullmatch(r"[A-Za-z0-9_-]+", language):
        raise FetchError("语言代码只能包含字母、数字、连字符或下划线。", "input")
    with redirect_stdout(sys.stderr):
        try:
            text = platforms.fetch_subtitles(video, language, chrome_profile)
            platforms.validate_srt_text(text)
            return text
        except FetchError as error:
            if error.kind == "input":
                raise
            caption_failure = platforms.safe_message(error)
            report(f"未取得有效平台字幕：{caption_failure}")
        report("尝试下载音频并使用本地 Whisper 转录。")
        try:
            hint = transcription_language(language)
            with tempfile.TemporaryDirectory(prefix="koesub-audio-") as scratch:
                media = platforms.download_audio(video, Path(scratch), chrome_profile)
                model = load_model(model_path)
                text = transcribe_to_srt(model, media, language=hint)
                count = platforms.validate_srt_text(text)
            report(f"字幕来源：本地 Whisper 转录；已验证 {count} 条字幕。")
            return text
        except (FetchError, OSError, ValueError, RuntimeError, KeyError,
                ImportError, subprocess.SubprocessError) as error:
            raise FetchError(
                f"平台字幕获取失败：{caption_failure}\n"
                f"本地转录失败：{platforms.safe_message(error)}", "transcription") from None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="koesub fetch",
        description="获取 YouTube/Bilibili 字幕；无有效字幕时自动下载音频并转录，SRT 输出到 stdout。",
    )
    parser.add_argument("url", metavar="URL", help="单个视频链接；Bilibili 可带 ?p=2。")
    parser.add_argument(
        "--lang", "-l", metavar="LANG",
        help="字幕语言及转录语言提示（如 ja、en、zh-CN）；默认平台原语言优先，转录自动检测。",
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_DIR, metavar="PATH",
        help="本地 .pt 模型或仅包含一个模型的目录；仅转录时加载。默认 $XDG_DATA_HOME/whisper/models；"
             "未设置时为 ~/.local/share/whisper/models。",
    )
    parser.add_argument("--chrome-profile", metavar="PROFILE", help="登录重试使用的 Chrome 配置。")
    args = parser.parse_args(argv)
    try:
        text = fetch_srt(args.url, language=args.lang, model_path=args.model_path,
                         chrome_profile=args.chrome_profile)
        sys.stdout.write(text)
        return 0
    except (FetchError, OSError, ValueError, RuntimeError, ImportError,
            KeyError, subprocess.SubprocessError) as error:
        report(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        report("已取消。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
