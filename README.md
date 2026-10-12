# koesub

Fetch subtitles from YouTube or Bilibili, transcribe local audio/video with
Whisper, and create black-background subtitled videos. Available as both a
Python library and a command-line tool.

## Requirements

- Python 3.11 or newer.
- FFmpeg and ffprobe on `PATH` for subtitle conversion, media decoding, duration
  checks, and video encoding.
- Node.js on `PATH` for YouTube extraction with the configured yt-dlp runtime.
- A local multilingual Whisper `.pt` checkpoint when transcription is needed.
  Models are not bundled with the package or downloaded automatically.

On macOS, install the external programs with:

```fish
brew install ffmpeg node
```

Transcription selects CUDA, MPS, or CPU according to PyTorch availability.
The video renderer currently requires macOS because it uses the system Hiragino
font at `/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc`.

## Installation

Run the published package from PyPI in uv's isolated tool environment:

```fish
uvx koesub --help
uvx koesub fetch "VIDEO_URL" > subtitles.srt
```

For a persistent CLI installation:

```fish
uv tool install koesub
koesub --help
```

To install the library into another virtual environment:

```fish
uv pip install koesub
```

Use `koesub COMMAND` or `python -m koesub COMMAND`. The public Python API is
under the `koesub` package. Installation includes yt-dlp, Whisper, PyTorch, and
Pillow. External programs and model checkpoints remain separate requirements.

For development from a source checkout, use `uv sync` and `uv run koesub`.

## Fetch subtitles from a URL

```fish
uvx koesub fetch "https://www.youtube.com/watch?v=YE7VzlLtp-4" > subtitles.srt
uvx koesub fetch "https://www.bilibili.com/video/BV12N4y1M7rh?p=2" --lang zh-CN > subtitles.srt
uvx koesub fetch "VIDEO_URL" --lang ja \
    --model-path ~/.local/share/whisper/models/large-v3-turbo.pt > subtitles.srt
```

`fetch` accepts one YouTube or Bilibili video URL and writes complete, validated
SRT content to stdout. Language/source reports, authentication retries, Whisper
progress, and errors go to stderr. It returns 0 on success, nonzero on failure,
and 130 on interruption. Failed fetches produce no SRT on stdout, although shell
redirection may create an empty destination file.

| Argument | Behavior |
| --- | --- |
| `URL` | Single YouTube/Bilibili video; Bilibili supports `?p=2`. |
| `--lang LANG`, `-l LANG` | Platform subtitle language and Whisper language hint. Default: prefer the original platform language, auto-detect during transcription. |
| `--model-path PATH` | Local `.pt` checkpoint, or a directory containing exactly one checkpoint. |
| `--chrome-profile PROFILE` | Chrome profile used during authentication retries. |
| `--help`, `-h` | Show command help. |

The default model directory is `$XDG_DATA_HOME/whisper/models`, or
`~/.local/share/whisper/models` when `XDG_DATA_HOME` is unset.

Platform captions are preferred, with manual captions preferred over automatic
captions in the same language; identifiable automatic translations, danmaku,
and chat are excluded. If valid captions cannot be obtained, including caption
selection, download, conversion, or validation failures, the command automatically
downloads audio and attempts local Whisper transcription. Audio-only formats are
preferred; a single media file containing audio is used when necessary. Whisper
is loaded only on this fallback path, and stderr identifies the successful source.

Each download starts anonymously. Explicit authentication errors trigger Safari,
then Chrome retries. Cookie values and signed URLs are sanitized in diagnostics;
cookies are not exported. Unsupported inputs are rejected before network access.
Temporary captions and audio are cleaned up on success, failure, or interruption.

There is no `--output-dir` and no transcription opt-in flag. The caller owns final
SRT persistence. Dialect/script hints such as `zh-CN`, `zh-Hans`, and `en-GB` map
to `zh` and `en` for Whisper. The command transcribes speech without translating
it, and recognition accuracy should be reviewed before relying on exact wording.

## Transcribe a local file

```fish
uvx koesub transcribe audio.mp3 --language ja > subtitles.srt
uvx koesub transcribe video.mp4 > subtitles.srt
```

`transcribe` accepts exactly one local audio/video file. Whisper uses FFmpeg to
decode its audio stream, so no separate audio-extraction step is needed. It writes
SRT to stdout and diagnostics to stderr; empty subtitles or processing failures
return a nonzero exit status without writing SRT.

Use `--language LANG` / `-l LANG` to set the audio language; the default is automatic
detection. `--model-path PATH` / `--model-dir PATH` select a local checkpoint or
single-checkpoint directory, with the same default model directory as `fetch`.

## Create a subtitled video

```fish
uvx koesub make-video audio.mp3 --language ja
uvx koesub make-video first.mp3 second.flac --output-dir ./outputs
uvx koesub make-video audio.mp3 --audio-codec copy --container mkv
```

`make-video` creates a 1280×720, 25 fps video with a black background and white
subtitles. It reads existing SRT from `OUTPUT_DIR/srt/INPUT_STEM.srt`, or transcribes
missing subtitles in memory. Batch jobs reuse one Whisper model. Duplicate input
stems are rejected before processing to prevent collisions within a batch.

Outputs are written to `OUTPUT_DIR/CONTAINER/INPUT_STEM.CONTAINER`; the default
directory is `./outputs` and the default format is H.264/AAC in MP4. Existing
target videos are overwritten. Use `--audio-codec`, `--video-codec`, and
`--container` to select available FFmpeg encoders and an output container
(`mp4`, `mkv`, `mov`, `avi`, or `ts`). Automatic transcription accepts the same
language/model options as `transcribe`.

The renderer requires non-overlapping captions and checks subtitle timing before
encoding. Platform subtitle overlaps are preserved by `fetch`, so some fetched
subtitles require editing before use by the renderer. Generated subtitles are
not automatically persisted by `make-video`.

## Python API

Fetch from a URL and receive an SRT string:

```python
from koesub.fetch import fetch_srt

srt = fetch_srt("https://www.youtube.com/watch?v=YE7VzlLtp-4", language="en")
```

Reuse a loaded model for local files:

```python
from koesub.transcription import load_model, transcribe_to_srt

model = load_model("/path/to/large-v3-turbo.pt")
first_srt = transcribe_to_srt(model, "first.mp3", language="ja")
second_srt = transcribe_to_srt(model, "second.flac", language="ja")
```

Library calls return strings and raise exceptions on failure. Callers control
subtitle persistence and model reuse. Calls are sequential: inference diagnostics
are redirected to stderr using process-wide stdout redirection.

## Development and building

The project uses the native `uv_build` backend. Tests and the project skill are
included in the source distribution; the wheel contains the `koesub` package,
CLI entrypoint, README metadata, and MIT license.

```fish
uv sync
uv run python -m unittest discover -s tests -v
uv build --no-sources --clear
uvx twine check dist/*
```

Build artifacts are written to `dist/` as a source distribution (`.tar.gz`) and
an installable wheel (`.whl`). Building does not publish the package. Network and
browser authentication tests use simulated services; they do not establish live
YouTube/Bilibili availability.

## Video subtitle analysis skill

The project skill is at `skills/video-subtitles/SKILL.md`, with UI metadata at
`skills/video-subtitles/agents/openai.yaml`. It captures koesub's SRT stdout as a
workspace artifact and guides timestamped analysis while identifying platform
captions or local Whisper transcription. The skill uses `uvx koesub` from PyPI
and does not require a local project checkout or virtual environment.

Invoke it explicitly by providing the skill's path and a video URL:

```text
Read and use /path/to/koesub/skills/video-subtitles/SKILL.md to fetch this video's subtitles and summarize its content with timestamps: VIDEO_URL
```

The repository `skills/` folder is not automatically a globally registered skill.
After installation in a supported discovery location, it can be invoked as
`$video-subtitles`. Its policy allows automatic selection by default. The skill
is included in the source archive and remains separate from wheel installation.

## License

koesub is released under the [MIT License](LICENSE). Third-party dependencies,
model checkpoints, and system fonts retain their own licenses.
