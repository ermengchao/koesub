"""Create subtitled videos, reusing existing SRT or transcribing in memory."""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .transcription import DEFAULT_MODEL_DIR, load_model, transcribe_to_srt


def seconds(value):
    match = re.fullmatch(r'(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})', value)
    if not match:
        raise ValueError(f'Invalid SRT timestamp: {value}')
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt(text, duration):
    """Validate subtitles before creating frames or starting an encoder."""
    text = text.lstrip('\ufeff').strip()
    if not text:
        raise ValueError('Empty subtitles')
    cues = []
    previous = 0.0
    for block in re.split(r'\n\s*\n', text):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError('Invalid SRT block: expected index, timestamps and text')
        timestamps = lines[1].split(' --> ')
        if len(timestamps) != 2:
            raise ValueError(f'Invalid SRT timing line: {lines[1]}')
        start, end = map(seconds, timestamps)
        # SRT rounds to milliseconds; its final timestamp can exceed the
        # exact media duration by half a millisecond.
        if not 0 <= start < end <= duration + 0.0005:
            raise ValueError(f'Subtitle interval outside media duration: {lines[1]}')
        end = min(end, duration)
        if start >= end:
            raise ValueError(f'Subtitle interval outside media duration: {lines[1]}')
        if start < previous - 0.001:
            raise ValueError(f'Overlapping or unordered subtitles: {lines[1]}')
        caption = '\n'.join(lines[2:])
        if not caption.strip():
            raise ValueError('Empty subtitle text')
        cues.append((start, end, caption))
        previous = end
    return cues


def frame(text, destination, font):
    im = Image.new('RGB', (1280, 720), 'black')
    draw = ImageDraw.Draw(im)
    lines = []
    for paragraph in text.splitlines():
        line = ''
        for char in paragraph:
            if draw.textlength(line + char, font=font) > 1160 and line:
                lines.append(line)
                line = ''
            line += char
        if line:
            lines.append(line)
    y = 360 - (len(lines) * 64) / 2
    for line in lines:
        draw.text((640, y), line, font=font, fill='white', anchor='mt')
        y += 64
    im.save(destination)


def encoder_settings(codec):
    encoder = {
        'h264': 'libx264', 'h265': 'libx265', 'hevc': 'libx265',
        'vp9': 'libvpx-vp9', 'av1': 'libsvtav1',
    }.get(codec, codec)
    listing = subprocess.check_output(['ffmpeg', '-v', 'error', '-encoders'], text=True)
    encoders = {}
    for line in listing.splitlines():
        match = re.match(r'\s*V\S{5}\s+(\S+)\s+(.+)', line)
        if match:
            name, description = match.groups()
            codec_match = re.search(r'\(codec ([^)]+)\)', description)
            encoders[name] = codec_match.group(1) if codec_match else name
    if encoder not in encoders:
        raise ValueError(f'Video encoder unavailable in this FFmpeg build: {encoder}')
    return encoder, encoders[encoder]


def render_video(audio, cues, target, duration, *, audio_codec, video_encoder, video_codec_name):
    font = ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc', 40)
    container = target.suffix[1:]
    video_options = ['-c:v', video_encoder]
    if video_encoder == 'libx264':
        video_options += ['-preset', 'veryfast', '-tune', 'stillimage', '-crf', '22']
    elif video_encoder == 'libx265':
        video_options += ['-preset', 'veryfast', '-crf', '22']

    with tempfile.TemporaryDirectory(prefix='koesub-frames-') as temp_dir:
        folder = Path(temp_dir)
        black = folder / 'black.png'
        frame('', black, font)
        playlist = []

        def add(path, length):
            if length > 0:
                playlist.extend([f"file '{path}'", f'duration {length:.6f}'])

        previous = 0.0
        for index, (start, end, caption) in enumerate(cues):
            image = folder / f'{index:03d}.png'
            frame(caption, image, font)
            add(black, start - previous)
            add(image, end - start)
            previous = end
        add(black, duration - previous)
        playlist.append(f"file '{black}'")
        listing = folder / 'frames.txt'
        listing.write_text('\n'.join(playlist) + '\n', encoding='utf-8')
        muxer = {'mkv': 'matroska', 'ts': 'mpegts'}.get(container, container)
        audio_options = ['-c:a', audio_codec]
        if audio_codec in {'aac', 'mp3', 'libmp3lame', 'opus', 'libopus', 'vorbis', 'libvorbis'}:
            audio_options += ['-b:a', '192k']
        container_options = ['-f', muxer]
        if container in {'mp4', 'mov'}:
            container_options += ['-movflags', '+faststart']
        subprocess.run([
            'ffmpeg', '-y', '-v', 'warning', '-f', 'concat', '-safe', '0',
            '-i', str(listing), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
            '-vf', 'fps=25', *video_options, '-pix_fmt', 'yuv420p', *audio_options,
            '-t', str(duration), *container_options, str(target),
        ], check=True)
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(target),
        ]))
        assert [stream['codec_type'] for stream in info['streams']] == ['video', 'audio']
        assert info['streams'][0]['codec_name'] == video_codec_name
        if audio_codec == 'copy':
            source_info = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name', '-of', 'json', str(audio),
            ]))
            assert info['streams'][1]['codec_name'] == source_info['streams'][0]['codec_name']
        assert info['streams'][0]['pix_fmt'] == 'yuv420p'
        assert abs(float(info['format']['duration']) - duration) < 0.1
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(target), '-f', 'null', '-'], check=True)
    print(f'DONE {target}: {len(cues)} cues, duration={duration:.3f}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Create black-background subtitled videos from audio. '
                    'Use existing SRT files, or transcribe missing subtitles in memory.'
    )
    parser.add_argument('audio', nargs='+', type=Path, metavar='AUDIO',
                        help='One or more input audio files (required).')
    parser.add_argument(
        '--output-dir', '-o', type=Path, default=Path.cwd() / 'outputs', metavar='DIR',
        help='Output directory and existing SRT lookup root (default: ./outputs).',
    )
    parser.add_argument(
        '--language', '-l', default=None, metavar='LANG',
        help='Language for automatic transcription (e.g. en, zh, ja). Default: auto-detect.',
    )
    parser.add_argument(
        '--model-path', '--model-dir', type=Path, default=DEFAULT_MODEL_DIR, metavar='PATH',
        help='Local .pt model file or directory containing one model; used only when SRT is missing '
             '(default: $XDG_DATA_HOME/whisper/models; ~/.local/share when unset).',
    )
    parser.add_argument(
        '--audio-codec', '--audio-format', default='aac', metavar='CODEC',
        help='FFmpeg audio encoder, or copy to preserve source audio (default: aac).',
    )
    parser.add_argument(
        '--video-codec', '--video-format', default='h264', metavar='CODEC',
        help='Video codec (h264, h265/hevc, vp9, av1) or FFmpeg encoder name (default: h264).',
    )
    parser.add_argument('--container', choices=('mp4', 'mkv', 'mov', 'avi', 'ts'), default='mp4',
                        help='Output container and file extension (default: mp4).')
    args = parser.parse_args(argv)
    audio_files = [audio.expanduser().resolve() for audio in args.audio]
    stems = set()
    for audio in audio_files:
        if not audio.is_file():
            parser.error(f'Audio file not found: {audio}')
        key = audio.stem.casefold()
        if key in stems:
            parser.error(f'Duplicate input stem would share subtitles or overwrite video: {audio.stem}')
        stems.add(key)

    output_dir = args.output_dir.expanduser().resolve()
    model = None
    try:
        video_encoder, video_codec_name = encoder_settings(args.video_codec)
        for audio in audio_files:
            duration = float(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1', str(audio),
            ]))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f'Invalid audio duration: {audio}')
            srt_path = output_dir / 'srt' / f'{audio.stem}.srt'
            if srt_path.exists():
                srt_text = srt_path.read_text(encoding='utf-8-sig')
                source = str(srt_path)
            else:
                if model is None:
                    model = load_model(args.model_path)
                srt_text = transcribe_to_srt(model, audio, language=args.language)
                source = f'generated subtitles for {audio}'
            try:
                cues = parse_srt(srt_text, duration)
            except ValueError as error:
                raise ValueError(f'Invalid subtitles ({source}): {error}') from error
            target = output_dir / args.container / f'{audio.stem}.{args.container}'
            target.parent.mkdir(parents=True, exist_ok=True)
            render_video(audio, cues, target, duration, audio_codec=args.audio_codec,
                         video_encoder=video_encoder, video_codec_name=video_codec_name)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
