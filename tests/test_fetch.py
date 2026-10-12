"""Fetch contracts: no network or real browser cookie reads."""

import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from koesub import cli, fetch, platforms

URL = "https://www.youtube.com/watch?v=YE7VzlLtp-4"
SRT = "1\n00:00:00,100 --> 00:00:01,000\nこんにちは\n\n"
ROOT = Path(__file__).resolve().parents[1]


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def test_platform_success_never_downloads_audio_or_loads_model(self):
        with patch.object(platforms, "fetch_subtitles", return_value=SRT), \
             patch.object(platforms, "download_audio") as download, \
             patch.object(fetch, "load_model") as load, \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            code = fetch.main([URL, "--model-path", "/missing/checkpoint.pt"])
        self.assertEqual(code, 0)
        self.assertEqual(self.stdout.getvalue(), SRT)
        load.assert_not_called()
        download.assert_not_called()

    def test_caption_failures_automatically_transcribe_and_clean_up(self):
        for kind in ("no_subtitles", "language", "conversion", "invalid", "auth", "terminal"):
            with self.subTest(kind=kind):
                downloaded = []

                def download(video, scratch, profile):
                    media = scratch / "audio.m4a"
                    media.write_bytes(b"fake audio")
                    downloaded.append(media)
                    print("downloader diagnostics")
                    return media

                model = object()
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(platforms, "fetch_subtitles",
                                  side_effect=platforms.FetchError("no captions", kind)), \
                     patch.object(platforms, "download_audio", side_effect=download), \
                     patch.object(fetch, "load_model", return_value=model) as load, \
                     patch.object(fetch, "transcribe_to_srt", return_value=SRT) as transcribe, \
                     contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = fetch.main([URL, "--model-path", "model.pt", "--chrome-profile", "Profile 1"])
                self.assertEqual(code, 0)
                self.assertEqual(stdout.getvalue(), SRT)
                self.assertIn("downloader diagnostics", stderr.getvalue())
                self.assertIn("本地 Whisper", stderr.getvalue())
                load.assert_called_once_with(Path("model.pt"))
                transcribe.assert_called_once_with(model, downloaded[0], language=None)
                self.assertFalse(downloaded[0].parent.exists())

    def test_invalid_caption_text_falls_back(self):
        with patch.object(platforms, "fetch_subtitles", return_value="<html>bad</html>"), \
             patch.object(platforms, "download_audio", return_value=Path("audio.wav")) as download, \
             patch.object(fetch, "load_model", return_value=object()), \
             patch.object(fetch, "transcribe_to_srt", return_value=SRT), \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(fetch.main([URL]), 0)
        download.assert_called_once()
        self.assertEqual(self.stdout.getvalue(), SRT)

    def test_language_aliases_are_whisper_hints(self):
        with contextlib.redirect_stderr(self.stderr):
            for code, expected in (("ja", "ja"), ("zh-CN", "zh"), ("zh-Hans", "zh"),
                                   ("en-GB", "en"), ("en_US", "en"), ("iw", "he")):
                with self.subTest(code=code):
                    self.assertEqual(fetch.transcription_language(code), expected)
            with self.assertRaises(platforms.FetchError):
                fetch.transcription_language("unknown")
        self.assertIsNone(fetch.transcription_language(None))

    def test_language_and_profile_are_forwarded(self):
        with patch.object(platforms, "fetch_subtitles",
                          side_effect=platforms.FetchError("none", "no_subtitles")) as captions, \
             patch.object(platforms, "download_audio", return_value=Path("audio.wav")) as download, \
             patch.object(fetch, "load_model", return_value=object()), \
             patch.object(fetch, "transcribe_to_srt", return_value=SRT) as transcribe, \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(fetch.main([URL, "-l", "zh-Hans", "--chrome-profile", "Profile 1"]), 0)
        self.assertEqual(captions.call_args.args[1:], ("zh-Hans", "Profile 1"))
        self.assertEqual(download.call_args.args[2], "Profile 1")
        self.assertEqual(transcribe.call_args.kwargs, {"language": "zh"})

    def test_both_paths_fail_with_empty_stdout_and_sanitized_error(self):
        with patch.object(platforms, "fetch_subtitles",
                          side_effect=platforms.FetchError("no captions", "no_subtitles")), \
             patch.object(platforms, "download_audio",
                          side_effect=platforms.FetchError("https://example.test/?token=SECRET", "auth")), \
             patch.object(fetch, "load_model") as load, \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(fetch.main([URL]), 1)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertNotIn("SECRET", self.stderr.getvalue())
        self.assertIn("平台字幕获取失败", self.stderr.getvalue())
        self.assertIn("本地转录失败", self.stderr.getvalue())
        load.assert_not_called()

    def test_inference_failure_or_invalid_srt_cleans_temporary_audio(self):
        for outcome in (RuntimeError("inference failed"), "", "bad", SRT.replace("1\n", "2\n", 1)):
            with self.subTest(outcome=outcome):
                paths = []

                def download(video, scratch, profile):
                    media = scratch / "audio.wav"
                    media.touch()
                    paths.append(media)
                    return media

                kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(platforms, "fetch_subtitles",
                                  side_effect=platforms.FetchError("none", "no_subtitles")), \
                     patch.object(platforms, "download_audio", side_effect=download), \
                     patch.object(fetch, "load_model", return_value=object()), \
                     patch.object(fetch, "transcribe_to_srt", **kwargs), \
                     contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(fetch.main([URL]), 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(paths[0].parent.exists())

    def test_bad_url_or_language_never_fetches(self):
        for args in (["https://example.test/video"], [URL, "--lang", "ja/invalid"]):
            with self.subTest(args=args), patch.object(platforms, "fetch_subtitles") as captions, \
                 patch.object(platforms, "download_audio") as download, \
                 contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
                self.assertEqual(fetch.main(args), 1)
                captions.assert_not_called()
                download.assert_not_called()

    def test_removed_or_extra_arguments_fail_before_fetch(self):
        for args in ([URL, URL], [URL, "--output-dir", "out"], [URL, "--transcribe-if-missing"]):
            with self.subTest(args=args), patch.object(fetch, "fetch_srt") as run, \
                 contextlib.redirect_stderr(self.stderr):
                with self.assertRaises(SystemExit) as error:
                    fetch.main(args)
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_keyboard_interrupt_has_no_stdout(self):
        with patch.object(fetch, "fetch_srt", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.assertEqual(fetch.main([URL]), 130)
        self.assertEqual(self.stdout.getvalue(), "")

    def test_dispatch_and_help_do_not_import_inference_dependencies(self):
        script = ('import sys; from koesub.cli import main; '
                  'main(["fetch", "--help"])')
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-path", result.stdout)
        script = ('import sys; from koesub import fetch; '
                  'from koesub.transcription import load_model, transcribe_to_srt; '
                  'assert "torch" not in sys.modules and "whisper" not in sys.modules '
                  'and "yt_dlp" not in sys.modules')
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(fetch, "main", return_value=0) as run:
            self.assertEqual(cli.main(["fetch", URL, "-l", "ja"]), 0)
            run.assert_called_once_with([URL, "-l", "ja"])


class BrowserAndPlatformTests(unittest.TestCase):
    def run_stages(self, outcomes):
        calls = []

        def attempt(browser, profile):
            calls.append((browser, profile))
            result = outcomes[len(calls) - 1]
            if isinstance(result, Exception):
                raise result
            return result

        with contextlib.redirect_stderr(io.StringIO()):
            try:
                result = platforms.with_browser_fallback(attempt, "Profile 1")
            except platforms.FetchError as error:
                result = error
        return calls, result

    def test_authentication_order_and_error_kind(self):
        calls, result = self.run_stages([
            platforms.FetchError("login required", "auth"),
            platforms.FetchError("cookie inaccessible", "cookie"), SRT,
        ])
        self.assertEqual([call[0] for call in calls], [None, "safari", "chrome"])
        self.assertEqual(result, SRT)
        calls, result = self.run_stages([platforms.FetchError("none", "no_subtitles")])
        self.assertEqual([call[0] for call in calls], [None])
        self.assertEqual(result.kind, "no_subtitles")

    def test_anonymous_success_or_non_auth_error_never_reads_cookies(self):
        calls, result = self.run_stages([SRT])
        self.assertEqual([call[0] for call in calls], [None])
        for kind in ("terminal", "language", "conversion", "invalid", "other"):
            with self.subTest(kind=kind):
                calls, result = self.run_stages([platforms.FetchError("failure", kind)])
                self.assertEqual([call[0] for call in calls], [None])
                self.assertEqual(result.kind, kind)

    def test_safari_no_captions_tries_chrome_but_network_failure_stops(self):
        calls, result = self.run_stages([
            platforms.FetchError("login required", "auth"),
            platforms.FetchError("none", "no_subtitles"), SRT,
        ])
        self.assertEqual([call[0] for call in calls], [None, "safari", "chrome"])
        calls, result = self.run_stages([
            platforms.FetchError("login required", "auth"),
            platforms.FetchError("429", "terminal"),
        ])
        self.assertEqual([call[0] for call in calls], [None, "safari"])
        self.assertEqual(result.kind, "terminal")

    def test_srt_validation_allows_platform_overlap_but_rejects_bad_cues(self):
        overlap = SRT + "2\n00:00:00,500 --> 00:00:02,000\nsecond\n\n"
        self.assertEqual(platforms.validate_srt_text(overlap), 2)
        for text in ("", "bad", SRT.replace("00:00:01,000", "00:00:00,100"),
                     SRT.replace("00:00:00,100", "00:61:00,100")):
            with self.subTest(text=text), self.assertRaises(platforms.FetchError):
                platforms.validate_srt_text(text)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_actual_vtt_and_bilibili_json_conversion(self):
        samples = [
            {"ext": "vtt", "data": "WEBVTT\n\n00:00:00.100 --> 00:00:01.000\nこんにちは\n\n"},
            {"ext": "json", "bilibili_json": True,
             "data": '{"body":[{"from":0.1,"to":1,"content":"こんにちは"}]}'},
        ]
        for sample in samples:
            with self.subTest(sample=sample["ext"]), tempfile.TemporaryDirectory() as scratch:
                track = platforms.Track("ja", "manual", False, (sample,))
                path = platforms.download_track(None, {}, track, Path(scratch))
                self.assertEqual(platforms.validate_srt(path), 1)
                self.assertEqual(path.read_text(), SRT)

    def test_actual_caption_attempt_returns_srt_without_model_or_persistence(self):
        import yt_dlp
        options_seen = []
        paths = []

        class YDL:
            def __init__(self, options):
                options_seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def extract_info(self, url, download=False):
                print("unexpected downloader stdout")
                return {"id": "test", "language": "ja", "subtitles": {
                    "ja": [{"ext": "srt", "data": SRT}]}}

        original = platforms.download_track

        def track(*args):
            path = original(*args)
            paths.append(path)
            return path

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(yt_dlp, "YoutubeDL", YDL), \
             patch.object(platforms, "download_track", side_effect=track), \
             patch.object(fetch, "load_model") as load, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(fetch.main([URL]), 0)
        self.assertEqual(stdout.getvalue(), SRT)
        self.assertIn("unexpected downloader stdout", stderr.getvalue())
        self.assertIn("人工字幕", stderr.getvalue())
        load.assert_not_called()
        self.assertFalse(paths[0].parent.exists())
        self.assertTrue(options_seen[0]["skip_download"])

    def test_audio_download_options_and_retry(self):
        import yt_dlp
        options_seen = []

        class YDL:
            def __init__(self, options):
                self.options = options
                options_seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            @property
            def cookiejar(self):
                return object()

            def extract_info(self, url, download=False):
                if len(options_seen) == 1:
                    raise RuntimeError("Sign in required to watch")
                assert download
                Path(self.prepare_filename({})).write_bytes(b"fake audio")
                return {"id": "test", "ext": "m4a"}

            def prepare_filename(self, info):
                return self.options["outtmpl"].replace("%(ext)s", "m4a")

        with tempfile.TemporaryDirectory() as scratch, \
             patch.object(yt_dlp, "YoutubeDL", YDL), \
             contextlib.redirect_stderr(io.StringIO()):
            media = platforms.download_audio(platforms.normalize_url(URL), Path(scratch), "Profile 1")
            self.assertEqual(media.read_bytes(), b"fake audio")
        self.assertFalse(media.exists())
        self.assertEqual(len(options_seen), 2)
        self.assertNotIn("cookiesfrombrowser", options_seen[0])
        self.assertEqual(options_seen[1]["cookiesfrombrowser"], ("safari", None))
        for options in options_seen:
            self.assertEqual(options["format"], "bestaudio/best")
            self.assertFalse(options["skip_download"])
            self.assertFalse(options["writesubtitles"])
            self.assertNotIn("dash", options["extractor_args"]["youtube"]["skip"])

    def test_bilibili_metadata_is_preserved(self):
        extractor = platforms.make_bilibili_extractor()
        response = {"code": 0, "data": {"subtitle": {"lan": "zh", "subtitles": [
            {"lan": "zh-CN", "type": 0, "subtitle_url": "//example.test/manual"},
            {"lan": "ai-zh", "type": 1, "ai_status": 2, "subtitle_url": "//example.test/auto"},
        ]}}}
        with patch.object(extractor, "_download_json", return_value=response):
            result = extractor._get_subtitles("BV12N4y1M7rh", 123)
        self.assertEqual(result["zh-CN"][0]["source_kind"], "manual")
        self.assertEqual(result["ai-zh"][0]["source_kind"], "automatic")


if __name__ == "__main__":
    unittest.main()
