"""Command dispatcher; import each command only when selected."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="koesub", description="Fetch, transcribe and render subtitles.")
    parser.add_argument("command", choices=("fetch", "transcribe", "make-video"))
    # Leave command options (including --help) to the selected CLI.
    args = parser.parse_args(argv[:1])
    if args.command == "fetch":
        from .fetch import main as command
    elif args.command == "transcribe":
        from .transcribe import main as command
    else:
        from .make_video import main as command
    return command(argv[1:])
