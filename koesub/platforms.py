#!/usr/bin/env python3
"""Platform subtitle selection, conversion, and authenticated media downloads."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urlsplit, urlencode


class FetchError(Exception):
    def __init__(self, message: str, kind: str = "other"):
        super().__init__(message)
        self.kind = kind


def safe_message(message: object) -> str:
    """Do not log signed URLs, cookie values, HTTP headers or ANSI escapes."""
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(message))
    text = re.sub(r"https?://[^\s<>\"']+", "[URL 已隐藏]", text)
    text = re.sub(
        r"(?im)\b(?:cookie|set-cookie|authorization|proxy-authorization)\s*[:=][^\r\n]*",
        "[认证信息已隐藏]", text)
    text = re.sub(
        r"(?i)\b(?:SESSDATA|SAPISID|APISID|SID|HSID|SSID|access_token|refresh_token|"
        r"po_token|token|signature|sig)\b\s*[=:]\s*[^\s,;]+",
        "[认证信息已隐藏]", text)
    return text


def report(message: object) -> None:
    print(safe_message(message), file=sys.stderr)


def is_auth_error(message: str) -> bool:
    text = message.lower()
    # Missing premium video formats are unrelated to subtitle authentication.
    if "format(s)" in text and "premium" in text:
        return False
    return bool(re.search(
        r"(?:sign\s*in|log\s*in|logged\s*in|login)[^\n]{0,100}"
        r"(?:required|require|available|confirm|access|view|watch)|"
        r"(?:required|require|available|confirm|access|only)[^\n]{0,100}"
        r"(?:sign\s*in|log\s*in|logged\s*in|login)|"
        r"authentication required|http error 401|需要登录|请先登录|未登录",
        text))


def is_terminal_error(message: str) -> bool:
    return bool(re.search(
        r"timed? out|timeout|429|too many requests|rate.?limit|"
        r"name or service not known|nodename nor servname|name resolution|"
        r"connection (?:refused|reset)|network is unreachable|"
        r"video (?:has been |was )?(?:removed|deleted)|"
        r"http error (?:404|5\d\d)|geo.?restrict|not available in your country|"
        r"operation not permitted|certificate verify failed",
        message, re.IGNORECASE))


class QuietLogger:
    """yt-dlp logger: retain only sanitized warnings/errors, ignore debug output."""
    def __init__(self):
        self.messages: list[str] = []
        self.auth_required = False
        self.terminal_warning = False

    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        self.auth_required |= is_auth_error(str(message))
        self.terminal_warning |= is_terminal_error(str(message))
        sanitized = safe_message(message)
        self.messages.append(sanitized)
        report(f"警告：{sanitized}")

    def error(self, message):
        self.auth_required |= is_auth_error(str(message))
        self.messages.append(safe_message(message))


@dataclass(frozen=True)
class Video:
    url: str
    platform: str
    part: int | None = None


def normalize_url(url: str) -> Video:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in ("https", "http") or parsed.username or parsed.password:
        raise FetchError("请输入不含认证信息的 http/https 视频链接。", "input")
    host = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)
    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"):
        if host.endswith("youtu.be"):
            video_id = parsed.path.strip("/")
        elif parsed.path.rstrip("/") == "/watch":
            video_id = query.get("v", [""])[0]
        else:
            match = re.fullmatch(r"/(?:shorts|live|embed)/([\w-]+)/?", parsed.path)
            video_id = match.group(1) if match else ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise FetchError("请输入具体的 YouTube 视频链接，不支持频道或播放列表链接。", "input")
        return Video(f"https://www.youtube.com/watch?v={video_id}", "youtube")
    if host in ("bilibili.com", "www.bilibili.com", "m.bilibili.com"):
        match = re.fullmatch(r"/video/(BV[A-Za-z0-9]{10}|av\d+)/?", parsed.path)
        if not match:
            raise FetchError("请输入 B 站 /video/BV… 或 /video/av… 链接；短链接请先展开。", "input")
        try:
            part = int(query.get("p", ["1"])[0])
            if part < 1:
                raise ValueError
        except ValueError:
            raise FetchError("B 站分 P 参数必须是正整数。", "input") from None
        return Video(f"https://www.bilibili.com/video/{match.group(1)}?{urlencode({'p': part})}",
                     "bilibili", part)
    raise FetchError("第一版只支持 YouTube 和 Bilibili 视频链接。", "input")


def language_key(code: str) -> str:
    code = re.sub(r"^(?:ai-|a-)", "", code.lower()).removesuffix("-orig")
    return {"zh-cn": "zh-hans", "zh-sg": "zh-hans", "zh-tw": "zh-hant",
            "zh-hk": "zh-hant", "zh-mo": "zh-hant"}.get(code, code)


def language_matches(code: str, wanted: str) -> bool:
    actual, target = language_key(code), language_key(wanted)
    return actual == target or ("-" not in target and actual.split("-")[0] == target)


@dataclass(frozen=True)
class Track:
    code: str
    kind: str  # manual / automatic / unknown (Bilibili without a source marker)
    original: bool
    formats: tuple[dict, ...]


def collect_tracks(info: dict, platform: str) -> list[Track]:
    tracks = []
    for field, base_kind in (("subtitles", "manual"), ("automatic_captions", "automatic")):
        for code, formats in (info.get(field) or {}).items():
            if code in ("danmaku", "live_chat") or not formats:
                continue
            kind = base_kind
            if platform == "bilibili":
                # yt-dlp puts Bilibili AI captions in 'subtitles', too.
                declared_kind = next((f.get("source_kind") for f in formats if f.get("source_kind")), None)
                kind = declared_kind or ("automatic" if code.startswith("ai-") else "unknown")
            native_formats = []
            original = code.endswith("-orig")
            for fmt in formats:
                if fmt.get("ext") not in ("srt", "vtt") and not fmt.get("bilibili_json"):
                    continue
                name = str(fmt.get("name") or "")
                if re.search(r"auto.?translat|自动翻译|翻译|翻譯|\bfrom\b", name, re.IGNORECASE):
                    continue
                params = parse_qs(urlsplit(fmt.get("url") or "").query)
                if "tlang" in params:
                    # Any requested target language is an auto-translation track.
                    continue
                if platform == "youtube" and field == "automatic_captions" and not original:
                    native_lang = params.get("lang", [None])[0]
                    if native_lang and not language_matches(code, native_lang):
                        continue
                original |= bool(re.search(r"\(original\)|原语言|原始语言", name, re.IGNORECASE))
                native_formats.append(fmt)
            if native_formats:
                tracks.append(Track(code, kind, original, tuple(native_formats)))
    # YouTube often provides both en and en-orig for the same native AI track.
    marked = {(language_key(t.code), t.kind) for t in tracks if t.original}
    return [t for t in tracks if t.original or (language_key(t.code), t.kind) not in marked]


def make_bilibili_extractor():
    """Preserve subtitle metadata lost by yt-dlp and fetch only the chosen body."""
    from yt_dlp.extractor.bilibili import BiliBiliIE
    from yt_dlp.utils import ExtractorError

    class SubtitleBiliBiliIE(BiliBiliIE):
        @classmethod
        def ie_key(cls):
            return BiliBiliIE.ie_key()

        def _get_subtitles(self, video_id, cid, aid=None):
            response = self._download_json(
                "https://api.bilibili.com/x/player/wbi/v2", video_id,
                query={"aid": aid, "cid": cid} if aid else {"bvid": video_id, "cid": cid},
                note="Extracting subtitle metadata", headers=self._HEADERS)
            code = response.get("code", 0)
            if code == -101:
                self.raise_login_required("Subtitles require login")
            if code in (-352, -412):
                raise ExtractorError("Subtitle API rate limit / risk control; try later", expected=True)
            if code:
                raise ExtractorError(f"Subtitle API error {code}: {response.get('message', '')}", expected=True)
            data = response.get("data") or {}
            if data.get("need_login_subtitle"):
                self.report_warning("Subtitles are only available when logged in")
            subtitle = data.get("subtitle") or {}
            self.subtitle_language = subtitle.get("lan") or None
            result = {}
            for item in subtitle.get("subtitles") or []:
                language, url = item.get("lan"), item.get("subtitle_url")
                if not language or not url:
                    continue
                # Numeric ai_type meanings differ between web and gRPC APIs;
                # never infer translation from that number alone.
                is_ai = item.get("type") == 1 or language.startswith("ai-") or bool(item.get("ai_status"))
                kind = "automatic" if is_ai else "manual" if item.get("type") == 0 else "unknown"
                result.setdefault(language, []).append({
                    "ext": "json", "url": "https:" + url if url.startswith("//") else url,
                    "bilibili_json": True, "source_kind": kind,
                    "name": item.get("lan_doc") or "",
                })
            return result

        def _real_extract(self, url):
            result = super()._real_extract(url)
            if getattr(self, "subtitle_language", None) and not result.get("language"):
                result["language"] = self.subtitle_language
            return result

    return SubtitleBiliBiliIE()


def select_track(info: dict, platform: str, wanted: str | None, logger: QuietLogger) -> Track:
    tracks = collect_tracks(info, platform)
    if not tracks:
        raise FetchError("无可用字幕轨道（弹幕、聊天和自动翻译不算字幕）。", "no_subtitles")
    rank = {"manual": 0, "unknown": 1, "automatic": 2}

    def best(items):
        return min(items, key=lambda t: (rank[t.kind], not t.original, t.code))

    if wanted:
        candidates = [t for t in tracks if language_matches(t.code, wanted)]
        if not candidates:
            available = ", ".join(sorted({t.code for t in tracks}))
            raise FetchError(f"没有指定语言 {wanted} 的字幕；可用语言：{available}", "language")
        exact = [t for t in candidates if language_key(t.code) == language_key(wanted)]
        return best(exact or candidates)
    known_languages = [info.get("language")]
    audio_languages = {f.get("language") for f in info.get("formats") or []
                       if f.get("acodec") not in (None, "none") and f.get("language")}
    if len(audio_languages) == 1:
        known_languages.extend(audio_languages)
    # Prefer explicit original track markers over an ambiguous dubbed audio language.
    originals = [t for t in tracks if t.original]
    original_languages = {language_key(t.code) for t in originals}
    if len(original_languages) == 1:
        original_language = next(iter(original_languages))
        return best([t for t in tracks if language_matches(t.code, original_language)])
    for language in known_languages:
        if language:
            candidates = [t for t in tracks if language_matches(t.code, language)]
            if candidates:
                return best(candidates)
    manual = [t for t in tracks if t.kind == "manual"]
    if len(manual) == 1:
        return manual[0]
    if len(originals) == 1:
        return originals[0]
    if len(tracks) == 1:
        return tracks[0]
    logger.warning("无法确认视频原语言，按语言代码排序选择；可用 --lang 明确指定。")
    return min(tracks, key=lambda t: (language_key(t.code), rank[t.kind], t.code))


SRT_CUE = re.compile(
    r"(?m)^(\d+)\n(\d{2,}:\d{2}:\d{2},\d{3}) --> "
    r"(\d{2,}:\d{2}:\d{2},\d{3})(?:[^\n]*)\n([^\n]+(?:\n(?!\n)[^\n]+)*)")


def timestamp_ms(value: str) -> int:
    hours, minutes, seconds, millis = map(int, re.split(r"[:,]", value))
    if minutes >= 60 or seconds >= 60:
        raise FetchError("字幕时间戳格式无效。", "invalid")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def validate_srt(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise FetchError(f"无法读取 UTF-8 字幕：{safe_message(error)}", "invalid") from None
    return validate_srt_text(text)


def validate_srt_text(text: str) -> int:
    """Validate platform or generated SRT; overlapping captions are allowed."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").strip()
    blocks = re.split(r"\n\s*\n", text) if text else []
    previous_start = -1
    for index, block in enumerate(blocks, 1):
        match = SRT_CUE.fullmatch(block)
        if not match or int(match[1]) != index:
            raise FetchError("SRT 包含无效条目或编号不连续。", "invalid")
        start, end = timestamp_ms(match[2]), timestamp_ms(match[3])
        if end <= start or start < previous_start or not match[4].strip():
            raise FetchError("SRT 时间轴或字幕内容无效。", "invalid")
        previous_start = start
    if not blocks:
        raise FetchError("字幕文件为空，没有有效条目。", "invalid")
    return len(blocks)


def download_track(ydl, info: dict, track: Track, scratch: Path) -> Path:
    from yt_dlp.networking import Request
    # Prefer platform SRT; only convert if SRT is not provided.
    fmt = next((f for f in track.formats if f["ext"] == "srt"), track.formats[0])
    raw = scratch / f"source.{fmt['ext']}"
    if "data" in fmt:
        raw.write_text(fmt["data"], encoding="utf-8")
    elif fmt.get("url"):
        headers = {**(info.get("http_headers") or {}), **(fmt.get("http_headers") or {})}
        with ydl.urlopen(Request(fmt["url"], headers=headers)) as response:
            # Subtitle files should remain small; never follow media playlists.
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise FetchError("字幕文件超过 32 MiB 限制。", "invalid")
        raw.write_bytes(data)
    else:
        raise FetchError("字幕轨道缺少数据或下载地址。", "invalid")
    if fmt.get("bilibili_json"):
        from yt_dlp.extractor.bilibili import BilibiliBaseIE
        try:
            body = json.loads(raw.read_text(encoding="utf-8-sig"))
            if not isinstance(body, dict) or not isinstance(body.get("body"), list):
                raise ValueError("missing subtitle body")
            text = BilibiliBaseIE().json2srt(body)
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise FetchError("B 站字幕 JSON 数据无效。", "invalid") from None
        target = scratch / "converted.srt"
        target.write_text(text, encoding="utf-8")
        return target
    if fmt["ext"] == "srt":
        return raw
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FetchError("字幕为 VTT，需要安装 FFmpeg 才能转换为 SRT。", "dependency")
    target = scratch / "converted.srt"
    try:
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(raw), "-c:s", "srt", str(target)],
            capture_output=True, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        raise FetchError("FFmpeg 字幕转换超时。", "conversion") from None
    if result.returncode:
        raise FetchError(f"FFmpeg 转换失败：{safe_message(result.stderr.strip())}", "conversion")
    return target


def downloader_options(logger, browser, chrome_profile, *, subtitles):
    node = shutil.which("node")
    options = {
        "logger": logger, "quiet": True, "no_warnings": False, "noprogress": True,
        "noplaylist": True, "playlist_items": "1", "cachedir": False,
        "socket_timeout": 20, "retries": 1, "extractor_retries": 1,
        "js_runtimes": {"node": {"path": node}} if node else {},
        "skip_download": subtitles, "writesubtitles": subtitles,
        "writeautomaticsub": subtitles, "ignore_no_formats_error": subtitles,
        "extractor_args": {"youtube": {"skip":
            ["translated_subs", "dash", "hls"] if subtitles else ["translated_subs"]}},
    }
    if browser:
        options["cookiesfrombrowser"] = (browser, chrome_profile if browser == "chrome" else None)
    return options


def load_cookies(ydl, browser):
    if browser:
        try:
            _ = ydl.cookiejar
        except Exception as error:
            raise FetchError(f"无法读取 {browser} Cookie：{safe_message(error)}", "cookie") from None


def extraction_error(error, logger):
    if isinstance(error, FetchError):
        return error
    if isinstance(error, OSError):
        return FetchError(f"本地文件或程序操作失败：{safe_message(error)}", "output")
    if is_terminal_error(str(error)):
        kind = "terminal"
    elif logger.auth_required or is_auth_error(str(error)):
        kind = "auth"
    else:
        kind = "other"
    return FetchError(safe_message(error), kind)


def with_browser_fallback(attempt_fn, chrome_profile=None):
    """Try anonymous first; access browsers only on authentication evidence."""
    failures = []
    last_kind = "other"
    for browser in (None, "safari", "chrome"):
        stage = browser or "匿名"
        report(f"尝试：{stage}")
        try:
            return attempt_fn(browser, chrome_profile)
        except FetchError as error:
            last_kind = error.kind
            failures.append(f"{stage}：{safe_message(error)}")
            if error.kind in ("terminal", "input", "language", "conversion", "invalid", "dependency", "output"):
                break
            if browser is None and error.kind != "auth":
                break
            report(f"{stage}未成功：{error}")
    raise FetchError("获取失败：\n" + "\n".join(failures), last_kind)


def attempt_subtitles(video, language, browser, chrome_profile):
    from yt_dlp import YoutubeDL
    logger = QuietLogger()
    try:
        with YoutubeDL(downloader_options(logger, browser, chrome_profile, subtitles=True)) as ydl:
            if video.platform == "bilibili":
                ydl.add_info_extractor(make_bilibili_extractor())
            load_cookies(ydl, browser)
            info = ydl.extract_info(video.url, download=False)
            if not info or info.get("_type") in ("playlist", "multi_video"):
                raise FetchError("未得到单个视频信息；不支持互动视频或合集。", "input")
            if logger.auth_required:
                raise FetchError("平台提示需要登录或身份验证才能完整获取字幕。", "auth")
            try:
                track = select_track(info, video.platform, language, logger)
            except FetchError as error:
                if error.kind == "no_subtitles" and logger.terminal_warning:
                    raise FetchError("提取字幕时发生网络、限流或访问错误。", "terminal") from None
                raise
            source = {"manual": "人工字幕", "automatic": "平台自动字幕",
                      "unknown": "平台字幕（来源未标明）"}[track.kind]
            report(f"选择语言 {track.code}，{source}；认证来源：{browser or '匿名'}。")
            with tempfile.TemporaryDirectory(prefix="koesub-subtitles-") as scratch:
                path = download_track(ydl, info, track, Path(scratch))
                count = validate_srt(path)
                text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
            report(f"已验证 {count} 条平台字幕。")
            return text.strip() + "\n\n"
    except Exception as error:
        raise extraction_error(error, logger) from None


def fetch_subtitles(video, language=None, chrome_profile=None):
    return with_browser_fallback(
        lambda browser, profile: attempt_subtitles(video, language, browser, profile), chrome_profile)


def attempt_audio(video, scratch, browser, chrome_profile):
    """Download audio-only when available, or a single muxed media file."""
    from yt_dlp import YoutubeDL
    logger = QuietLogger()
    folder = scratch / (browser or "anonymous")
    folder.mkdir(parents=True, exist_ok=True)
    options = downloader_options(logger, browser, chrome_profile, subtitles=False)
    options.update({"format": "bestaudio/best", "outtmpl": str(folder / "audio.%(ext)s")})
    try:
        with YoutubeDL(options) as ydl:
            load_cookies(ydl, browser)
            info = ydl.extract_info(video.url, download=True)
            if not info or info.get("_type") in ("playlist", "multi_video"):
                raise FetchError("未得到单个视频音频。", "input")
            audio = Path(ydl.prepare_filename(info)).resolve()
            if audio.parent != folder.resolve() or not audio.is_file() or audio.stat().st_size == 0:
                raise FetchError("未下载到有效的音频文件。", "output")
            report(f"音频下载完成；认证来源：{browser or '匿名'}。")
            return audio
    except Exception as error:
        raise extraction_error(error, logger) from None


def download_audio(video, scratch, chrome_profile=None):
    return with_browser_fallback(
        lambda browser, profile: attempt_audio(video, scratch, browser, profile), chrome_profile)
