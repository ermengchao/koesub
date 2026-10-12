"""Behavior tests; no network requests or actual browser cookie reads."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from koesub import platforms as fs


SRT = "1\n00:00:01,000 --> 00:00:02,500\n第一句话\n\n2\n00:00:03,000 --> 00:00:04,000\nSame words\n\n"


def fmt(ext="srt", **kwargs):
    return {"ext": ext, "data": SRT, **kwargs}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.log = fs.QuietLogger()

    def choose(self, info, platform="youtube", lang=None):
        return fs.select_track(info, platform, lang, self.log)

    def test_original_audio_before_other_language_manual(self):
        info = {"language": "ja", "subtitles": {"en": [fmt()]},
                "automatic_captions": {"ja-orig": [fmt()]}}
        self.assertEqual(self.choose(info).code, "ja-orig")

    def test_manual_before_automatic_same_language(self):
        info = {"language": "en", "subtitles": {"en": [fmt()]},
                "automatic_captions": {"en-orig": [fmt()]}}
        self.assertEqual(self.choose(info).kind, "manual")

    def test_original_marker_before_dubbed_audio(self):
        info = {"language": "en", "subtitles": {"en": [fmt()], "ja": [fmt()]},
                "automatic_captions": {"ja-orig": [fmt()]}}
        self.assertEqual(self.choose(info).code, "ja")

    def test_unique_manual_without_original_metadata(self):
        info = {"subtitles": {"ja": [fmt()]}, "automatic_captions": {"en": [fmt()]}}
        self.assertEqual(self.choose(info).code, "ja")

    def test_ambiguity_uses_code_and_warns(self):
        self.assertEqual(self.choose({"subtitles": {"ja": [fmt()], "en": [fmt()]}}).code, "en")
        self.assertTrue(any("无法确认" in m for m in self.log.messages))

    def test_explicit_language_missing_is_failure(self):
        with self.assertRaises(fs.FetchError) as error:
            self.choose({"subtitles": {"ja": [fmt()]}}, lang="en")
        self.assertEqual(error.exception.kind, "language")
        self.assertIn("ja", str(error.exception))

    def test_explicit_language_matches_original_and_chinese_aliases(self):
        self.assertEqual(self.choose({"automatic_captions": {"ja-orig": [fmt()]}}, lang="ja").code,
                         "ja-orig")
        self.assertEqual(self.choose({"subtitles": {"zh-Hans": [fmt()]}}, lang="zh-CN").code,
                         "zh-Hans")

    def test_region_is_not_silently_changed(self):
        with self.assertRaises(fs.FetchError):
            self.choose({"subtitles": {"en-US": [fmt()]}}, lang="en-GB")

    def test_danmaku_and_chat_are_not_subtitles(self):
        with self.assertRaises(fs.FetchError) as error:
            self.choose({"subtitles": {"danmaku": [{"ext": "xml"}], "live_chat": [fmt()]}})
        self.assertEqual(error.exception.kind, "no_subtitles")

    def test_translations_removed(self):
        info = {"automatic_captions": {
            "en-orig": [fmt(url="https://example.test/caption?lang=en")],
            "en": [fmt(url="https://example.test/caption?lang=en")],
            "ja": [fmt(url="https://example.test/caption?lang=en&tlang=ja")],
            "fr": [fmt(name="French (auto-translated)")],
        }}
        self.assertEqual([t.code for t in fs.collect_tracks(info, "youtube")], ["en-orig"])

    def test_bilibili_ai_is_automatic_and_other_source_is_unknown(self):
        info = {"subtitles": {"zh-CN": [fmt()], "ai-zh": [fmt()], "danmaku": [{"ext": "xml"}]}}
        tracks = fs.collect_tracks(info, "bilibili")
        self.assertEqual({t.code: t.kind for t in tracks}, {"zh-CN": "unknown", "ai-zh": "automatic"})

    def test_bilibili_metadata_selects_manual_and_excludes_translation(self):
        info = {"language": "zh", "subtitles": {
            "zh-CN": [fmt(source_kind="manual")],
            "ai-zh": [fmt(source_kind="automatic")],
            "ai-en": [fmt(source_kind="automatic", name="英语（自动翻译）")],
        }}
        self.assertEqual(self.choose(info, "bilibili").kind, "manual")
        self.assertEqual(len(fs.collect_tracks(info, "bilibili")), 2)
