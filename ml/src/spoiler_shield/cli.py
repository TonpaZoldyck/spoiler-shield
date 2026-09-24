"""Command-line entry point: ``spoiler-shield <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spoiler-shield", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    spike = sub.add_parser(
        "spike-model",
        help="Build the Phase 0 stand-in model: MiniLM-L6 architecture, random weights, int8 ONNX.",
    )
    spike.add_argument("--out", default="../extension/public/models/spike", help="Output directory")
    spike.add_argument("--seed", type=int, default=0)

    sub.add_parser("version", help="Print the package version.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "version":
        from spoiler_shield import __version__

        print(__version__)
        return 0
    if args.command == "spike-model":
        # Imported lazily: needs the heavy 'train' extra.
        from spoiler_shield.export.spike import build_spike_model

        build_spike_model(out_dir=args.out, seed=args.seed)
        return 0
    return 1  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":
    sys.exit(main())
