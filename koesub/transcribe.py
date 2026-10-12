"""Single-file CLI: SRT on stdout, diagnostics on stderr."""

import argparse
import subprocess
import sys
from pathlib import Path

from .transcription import DEFAULT_MODEL_DIR, load_model, transcribe_to_srt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Transcribe one audio or video file with Whisper; write SRT to stdout.'
    )
    parser.add_argument('media', type=Path, metavar='MEDIA', help='Input media file.')
    parser.add_argument(
        '--language', '-l', default=None, metavar='LANG',
        help='Audio language code (e.g. en, zh, ja). Default: auto-detect.',
    )
    parser.add_argument(
        '--model-path', '--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
        metavar='PATH',
        help='Local .pt model file or directory containing one model '
             '(default: $XDG_DATA_HOME/whisper/models; ~/.local/share when unset).',
    )
    args = parser.parse_args(argv)
    media = args.media.expanduser().resolve()
    if not media.is_file():
        parser.error(f'Media file not found: {media}')
    try:
        model = load_model(args.model_path)
        srt = transcribe_to_srt(model, media, language=args.language)
        if not srt.strip():
            raise RuntimeError(f'Empty subtitles: {media}')
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    sys.stdout.write(srt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
