"""Contract tests for CLI output, model reuse and subtitle selection."""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from koesub import make_video, transcribe, transcription

SRT = '1\n00:00:00,100 --> 00:00:01,000\nこんにちは\n\n'
RESULT = {'segments': [{
    'start': -0.5, 'end': 3.0, 'text': 'こんにちは',
    'words': [{'word': 'こんにちは', 'start': -0.5, 'end': 3.0}],
}]}
ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    device = SimpleNamespace(type='cpu')

    def __init__(self, result=RESULT):
        self.result = result
        self.calls = []

    def transcribe(self, path, **kwargs):
        print('Whisper progress')
        self.calls.append((path, kwargs))
        return copy.deepcopy(self.result)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.media = self.root / 'sample.mp3'
        self.media.touch()
        self.out = self.root / 'outputs'
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def test_import_has_no_execution_side_effects(self):
        script = (
            'import sys; sys.dont_write_bytecode = True; '
            f'sys.path.insert(0, {str(ROOT)!r}); '
            'sys.argv = ["caller", "--not-a-script-argument"]; '
            'from unittest.mock import patch; '
            '\nwith patch("subprocess.run", side_effect=AssertionError("subprocess")), '
            'patch("subprocess.check_output", side_effect=AssertionError("subprocess")), '
            'patch("pathlib.Path.mkdir", side_effect=AssertionError("mkdir")):\n'
            ' from koesub import transcription, transcribe, make_video\n'
            ' assert "torch" not in sys.modules and "whisper" not in sys.modules\n'
        )
        result = subprocess.run([sys.executable, '-c', script], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')
        self.assertEqual(list(self.root.iterdir()), [self.media])

    def test_cli_srt_and_diagnostics_are_separate(self):
        model = FakeModel()
        with patch.object(transcribe, 'load_model', return_value=model), \
             patch.object(transcription.subprocess, 'check_output',
                          return_value=json.dumps({'format': {'duration': '2'}}).encode()), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            code = transcribe.main([str(self.media), '-l', 'ja'])
        self.assertEqual(code, 0)
        text = self.stdout.getvalue()
        self.assertEqual(text, '1\n00:00:00,000 --> 00:00:02,000\nこんにちは\n\n')
        self.assertEqual(len(make_video.parse_srt(text, 2)), 1)
        self.assertIn('Whisper progress', self.stderr.getvalue())
        self.assertEqual(model.calls[0][1]['language'], 'ja')
        self.assertFalse(model.calls[0][1]['fp16'])
        self.assertFalse(self.out.exists())

    def test_cli_failure_or_empty_result_leaves_stdout_empty(self):
        for error in (RuntimeError('inference failed'), None):
            with self.subTest(error=error):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(transcribe, 'load_model', return_value=object()), \
                     patch.object(transcribe, 'transcribe_to_srt',
                                  side_effect=error, return_value=''), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    code = transcribe.main([str(self.media)])
                self.assertEqual(code, 1)
                self.assertEqual(stdout.getvalue(), '')
                self.assertIn('Error:', stderr.getvalue())

    def test_cli_rejects_multiple_inputs_and_removed_output_option(self):
        for args in ([str(self.media), str(self.media)],
                     [str(self.media), '--output-dir', str(self.out)]):
            with self.subTest(args=args), patch.object(transcribe, 'load_model') as load, \
                 redirect_stdout(self.stdout), redirect_stderr(self.stderr):
                with self.assertRaises(SystemExit) as error:
                    transcribe.main(args)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()

    def test_library_empty_result_raises_without_writing_files(self):
        with patch.object(transcription.subprocess, 'check_output',
                          return_value=b'{"format":{"duration":"2"}}'), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            with self.assertRaisesRegex(RuntimeError, 'Empty subtitles'):
                transcription.transcribe_to_srt(FakeModel({'segments': []}), self.media)
        self.assertEqual(self.stdout.getvalue(), '')
        self.assertEqual(list(self.root.iterdir()), [self.media])

    def test_generated_srt_rounding_is_accepted_and_clamped(self):
        duration = 1.999813
        with patch.object(transcription.subprocess, 'check_output',
                          return_value=json.dumps({'format': {'duration': str(duration)}}).encode()), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            text = transcription.transcribe_to_srt(FakeModel(), self.media)
        self.assertEqual(text, '1\n00:00:00,000 --> 00:00:02,000\nこんにちは\n\n')
        self.assertEqual(make_video.parse_srt(text, duration), [(0.0, duration, 'こんにちは')])

    def test_generated_srt_filters_zero_duration_cues_and_renumbers(self):
        # Cover segment-only results and word-timestamp results, including
        # positive intervals which collapse only when rounded to milliseconds.
        for with_words in (False, True):
            with self.subTest(with_words=with_words):
                intervals = [(0.0, 0.0), (0.1, 0.9), (1.0001, 1.0004),
                             (1.2, 1.8), (2.1, 2.5)]
                segments = []
                for start, end in intervals:
                    segment = {'start': start, 'end': end, 'text': 'test'}
                    if with_words:
                        segment['words'] = [{'word': 'test', 'start': start, 'end': end}]
                    segments.append(segment)
                with patch.object(transcription.subprocess, 'check_output',
                                  return_value=b'{"format":{"duration":"2"}}'), \
                     redirect_stdout(self.stdout), redirect_stderr(self.stderr):
                    text = transcription.transcribe_to_srt(FakeModel({'segments': segments}), self.media)
                self.assertEqual(text, '1\n00:00:00,100 --> 00:00:00,900\ntest\n\n'
                                       '2\n00:00:01,200 --> 00:00:01,800\ntest\n\n')
                self.assertEqual(len(make_video.parse_srt(text, 2)), 2)

    def test_generated_srt_preserves_zero_duration_words_in_valid_cue(self):
        result = {'segments': [{'start': 0.1, 'end': 0.9, 'text': 'test!', 'words': [
            {'word': 'test', 'start': 0.1, 'end': 0.9},
            {'word': '!', 'start': 0.9, 'end': 0.9},
        ]}]}
        with patch.object(transcription.subprocess, 'check_output',
                          return_value=b'{"format":{"duration":"2"}}'), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            text = transcription.transcribe_to_srt(FakeModel(result), self.media)
        self.assertEqual(make_video.parse_srt(text, 2), [(0.1, 0.9, 'test!')])

    def test_cli_all_cues_filtered_fails_without_stdout(self):
        for start, end in ((2.1, 2.5), (1.0001, 1.0004)):
            with self.subTest(start=start, end=end):
                result = {'segments': [{'start': start, 'end': end, 'text': 'test',
                                       'words': [{'word': 'test', 'start': start, 'end': end}]}]}
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(transcribe, 'load_model', return_value=FakeModel(result)), \
                     patch.object(transcription.subprocess, 'check_output',
                                  return_value=b'{"format":{"duration":"2"}}'), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    code = transcribe.main([str(self.media)])
                self.assertEqual(code, 1)
                self.assertEqual(stdout.getvalue(), '')
                self.assertIn('Empty subtitles', stderr.getvalue())

    def test_model_directory_resolution(self):
        with self.assertRaisesRegex(ValueError, 'No .pt model'):
            transcription.load_model(self.root)
        (self.root / 'a.pt').touch()
        (self.root / 'b.pt').touch()
        with self.assertRaisesRegex(ValueError, 'Multiple model files'):
            transcription.load_model(self.root)

    def test_mps_alignment_wrapper_is_not_nested(self):
        import torch
        import whisper
        import whisper.timing as timing
        checkpoint = self.root / 'model.pt'
        checkpoint.touch()
        with patch.object(torch.cuda, 'is_available', return_value=False), \
             patch.object(torch.backends.mps, 'is_available', return_value=True), \
             patch.object(torch, 'set_num_threads'), \
             patch.object(whisper, 'load_model', return_value=object()), \
             patch.object(timing, 'dtw', return_value='aligned') as original, \
             redirect_stderr(self.stderr):
            original._koesub_cpu_alignment = False
            transcription.load_model(checkpoint)
            wrapper = timing.dtw
            transcription.load_model(checkpoint)
            self.assertIs(timing.dtw, wrapper)
            matrix = SimpleNamespace(cpu=lambda: 'cpu matrix')
            self.assertEqual(wrapper(matrix), 'aligned')
            original.assert_called_once_with('cpu matrix')

    def run_video(self, files, existing=()):
        if existing:
            (self.out / 'srt').mkdir(parents=True)
            for media in existing:
                (self.out / 'srt' / f'{media.stem}.srt').write_text(SRT, encoding='utf-8')
        model = object()
        with patch.object(make_video, 'encoder_settings', return_value=('libx264', 'h264')), \
             patch.object(make_video.subprocess, 'check_output', return_value=b'2'), \
             patch.object(make_video, 'load_model', return_value=model) as load, \
             patch.object(make_video, 'transcribe_to_srt', return_value=SRT) as generate, \
             patch.object(make_video, 'render_video') as render, \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            code = make_video.main([*(str(media) for media in files), '-o', str(self.out), '-l', 'ja'])
        return code, load, generate, render, model

    def test_all_existing_srt_skips_model(self):
        code, load, generate, render, _ = self.run_video([self.media], [self.media])
        self.assertEqual(code, 0)
        load.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(render.call_count, 1)

    def test_missing_srt_loads_once_for_batch_and_stays_in_memory(self):
        other = self.root / 'other.flac'
        other.touch()
        code, load, generate, render, model = self.run_video([self.media, other])
        self.assertEqual(code, 0)
        load.assert_called_once()
        self.assertEqual(generate.call_count, 2)
        for call in generate.call_args_list:
            self.assertIs(call.args[0], model)
            self.assertEqual(call.kwargs, {'language': 'ja'})
        self.assertEqual(render.call_count, 2)
        self.assertFalse((self.out / 'srt').exists())

    def test_mixed_batch_only_transcribes_missing(self):
        other = self.root / 'other.mp3'
        other.touch()
        code, load, generate, render, model = self.run_video([self.media, other], [self.media])
        self.assertEqual(code, 0)
        load.assert_called_once()
        generate.assert_called_once_with(model, other, language='ja')
        self.assertEqual(render.call_count, 2)
        self.assertEqual(list((self.out / 'srt').iterdir()), [self.out / 'srt' / 'sample.srt'])

    def test_invalid_existing_srt_is_not_retranscribed(self):
        (self.out / 'srt').mkdir(parents=True)
        (self.out / 'srt' / 'sample.srt').write_text('broken', encoding='utf-8')
        code, load, generate, render, _ = self.run_video([self.media])
        self.assertEqual(code, 1)
        load.assert_not_called()
        generate.assert_not_called()
        render.assert_not_called()
        self.assertIn('Invalid subtitles', self.stderr.getvalue())

    def test_duplicate_stems_fail_before_external_work(self):
        other = self.root / 'sample.flac'
        other.touch()
        with patch.object(make_video, 'encoder_settings') as encoders, \
             patch.object(make_video, 'load_model') as load, \
             patch.object(make_video, 'render_video') as render, \
             redirect_stderr(self.stderr):
            with self.assertRaises(SystemExit) as error:
                make_video.main([str(self.media), str(other), '-o', str(self.out)])
        self.assertEqual(error.exception.code, 2)
        encoders.assert_not_called()
        load.assert_not_called()
        render.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_subtitle_validation_rejects_invalid_intervals(self):
        samples = ['', 'broken', SRT.replace('01,000', '03,000'),
                   SRT.replace('00,100', '01,000'), SRT + SRT]
        for text in samples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                make_video.parse_srt(text, 2)
        self.assertEqual(make_video.parse_srt('\ufeff' + SRT.replace('\n', '\r\n'), 2),
                         [(0.1, 1.0, 'こんにちは')])

    def test_subtitle_rounding_tolerance_rejects_genuine_overruns(self):
        text = '1\n00:00:00,100 --> 00:00:02,000\ntest\n\n'
        for duration in (1.999813, 1.9995):
            with self.subTest(duration=duration):
                self.assertEqual(make_video.parse_srt(text, duration), [(0.1, duration, 'test')])
        for duration in (1.999499, 1.999, 1.9):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                make_video.parse_srt(text, duration)
        outside = '1\n00:00:02,000 --> 00:00:02,000\ntest\n\n'
        with self.assertRaises(ValueError):
            make_video.parse_srt(outside, 1.9998)


if __name__ == '__main__':
    unittest.main()
