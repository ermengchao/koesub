"""Reusable Whisper transcription returning SRT without writing files.

Calls are sequential: console output is redirected to stderr while loading or
transcribing. Callers own model reuse and subtitle persistence.
"""

import io
import json
import math
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from functools import wraps
from pathlib import Path

DEFAULT_MODEL_DIR = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'whisper/models'


def load_model(model_path=DEFAULT_MODEL_DIR):
    """Load a local .pt checkpoint, or the sole checkpoint in a directory."""
    model_path = Path(model_path).expanduser()
    if model_path.is_file() or model_path.suffix == '.pt':
        checkpoint = model_path
    elif model_path.is_dir():
        checkpoints = sorted(path for path in model_path.glob('*.pt') if path.is_file())
        if not checkpoints:
            raise ValueError(f'No .pt model files found in: {model_path}')
        if len(checkpoints) > 1:
            names = ', '.join(path.name for path in checkpoints)
            raise ValueError(
                f'Multiple model files found: {names}. '
                'Use --model-path to specify a particular .pt file.'
            )
        checkpoint = checkpoints[0]
    else:
        raise ValueError(f'Model path not found: {model_path}')

    # Lazy imports keep CLI help and existing-SRT video jobs lightweight.
    with redirect_stdout(sys.stderr):
        import torch
        import whisper
        import whisper.timing as timing

        torch.set_num_threads(os.cpu_count() or 1)
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f'Loading openai-whisper {whisper.__version__}, device={device}', flush=True)
        model = whisper.load_model(str(checkpoint), device=device)
        if device == 'mps' and not getattr(timing.dtw, '_koesub_cpu_alignment', False):
            original_dtw = timing.dtw

            @wraps(original_dtw)
            def cpu_dtw(matrix):
                # Metal does not implement the float64 cast in Whisper's DTW.
                return original_dtw(matrix.cpu())

            cpu_dtw._koesub_cpu_alignment = True
            timing.dtw = cpu_dtw
    return model


def transcribe_to_srt(model, media, *, language=None) -> str:
    """Transcribe one media file with an already-loaded model and return SRT."""
    media = Path(media).expanduser().resolve()
    if not media.is_file():
        raise ValueError(f'Media file not found: {media}')
    duration = float(json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'json', str(media),
    ]))['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid media duration: {media}')

    started_at = time.monotonic()
    with redirect_stdout(sys.stderr):
        from whisper.utils import WriteSRT

        print(f'START {media.name}', flush=True)
        result = model.transcribe(
            str(media), language=language, task='transcribe', verbose=True,
            fp16=(model.device.type in ('cuda', 'mps')), beam_size=5,
            condition_on_previous_text=False, word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )

    for segment in result['segments']:
        segment['start'] = max(0, min(segment['start'], duration))
        segment['end'] = max(segment['start'], min(segment['end'], duration))
        for word in segment.get('words', []):
            word['start'] = max(0, min(word['start'], duration))
            word['end'] = max(word['start'], min(word['end'], duration))

    buffer = io.StringIO()
    subtitles = WriteSRT('').iterate_result(result, {
        'max_line_width': 32, 'max_line_count': None, 'highlight_words': False,
    })
    # Filter final cues after millisecond formatting, preserving zero-duration
    # words (such as punctuation) within otherwise valid captions.
    subtitles = ((start, end, text) for start, end, text in subtitles if start != end)
    for index, (start, end, text) in enumerate(subtitles, start=1):
        print(f'{index}\n{start} --> {end}\n{text}\n', file=buffer)
    srt = buffer.getvalue()
    if not srt.strip():
        raise RuntimeError(f'Empty subtitles: {media}')
    print(json.dumps({
        'source': str(media), 'seconds': duration,
        'elapsed_seconds': round(time.monotonic() - started_at, 1),
        'device': model.device.type, 'segments': len(result['segments']),
    }, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
    return srt
