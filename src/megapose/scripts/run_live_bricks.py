"""Command-line entry point for live brick pose output."""

# Standard Library
import argparse
import sys
from pathlib import Path

# MegaPose
from megapose.live_bricks import load_config, run_live
from megapose.utils.logging import set_logging_level


def main() -> None:
    parser = argparse.ArgumentParser(description="Output 2x4/2x2/1x2 brick poses as JSONL")
    parser.add_argument("--config", required=True, type=Path, help="Live pipeline YAML config")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Write poses to this file instead of standard output",
    )
    args = parser.parse_args()

    set_logging_level("warning")
    config = load_config(args.config)
    if args.output_jsonl is None:
        run_live(config, sys.stdout)
    else:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("a", encoding="utf-8") as output_stream:
            run_live(config, output_stream)


if __name__ == "__main__":
    main()
