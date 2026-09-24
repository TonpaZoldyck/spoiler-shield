"""Command-line entry point: ``spoiler-shield <command>``."""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spoiler-shield", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the package version.")

    spike = sub.add_parser(
        "spike-model",
        help="Build the Phase 0 stand-in model: MiniLM-L6 architecture, random weights, int8 ONNX.",
    )
    spike.add_argument("--out", default="../extension/public/models/spike", help="Output directory")
    spike.add_argument("--seed", type=int, default=0)

    prep = sub.add_parser("prepare", help="Convert raw datasets into one examples.jsonl file.")
    prep.add_argument("--goodreads", type=Path, help="goodreads_reviews_spoiler.json(.gz)")
    prep.add_argument("--imdb", type=Path, help="IMDB_reviews.json")
    prep.add_argument("--out", type=Path, default=Path("../data/processed/examples.jsonl"))

    base = sub.add_parser("baselines", help="Fit the baselines and write an evaluation report.")
    base.add_argument("--data", type=Path, default=Path("../data/processed/examples.jsonl"))
    base.add_argument("--out", type=Path, default=Path("../docs/results/phase1-baselines.md"))
    base.add_argument("--target-recall", type=float, default=0.9)
    base.add_argument(
        "--max-per-split", type=int, default=None, help="Deterministic subsample per split"
    )
    base.add_argument("--split", choices=["val", "test"], default="val")
    base.add_argument("--release", action="store_true", help="Required to report on the test split")
    return parser


def _prepare(args: argparse.Namespace) -> int:
    from spoiler_shield.data.schema import write_jsonl
    from spoiler_shield.data.sources import goodreads_examples, imdb_examples

    if not args.goodreads and not args.imdb:
        print("prepare: give --goodreads and/or --imdb", file=sys.stderr)
        return 2
    streams = []
    if args.goodreads:
        streams.append(goodreads_examples(args.goodreads))
    if args.imdb:
        streams.append(imdb_examples(args.imdb))
    counts: Counter[tuple[str, str]] = Counter()

    def counted():
        for ex in itertools.chain(*streams):
            counts[(ex.source, str(ex.label))] += 1
            yield ex

    n = write_jsonl(args.out, counted())
    print(f"wrote {n} examples to {args.out}")
    for (source, label), c in sorted(counts.items()):
        print(f"  {source:10s} label={label:5s} {c}")
    return 0


def _baselines(args: argparse.Namespace) -> int:
    from spoiler_shield.baselines import KeywordRules, TfidfLogReg
    from spoiler_shield.data.schema import Example, read_jsonl
    from spoiler_shield.data.splits import Split, assign_splits, subsample
    from spoiler_shield.eval.report import render_markdown, run_ladder

    splits: dict[Split, list[Example]] = assign_splits(read_jsonl(args.data))
    if args.max_per_split:
        splits = {k: subsample(v, args.max_per_split) for k, v in splits.items()}
    reports = run_ladder(
        [KeywordRules(), TfidfLogReg()],
        splits,
        target_recall=args.target_recall,
        report_split=args.split,
        allow_test=args.release,
    )
    sizes = ", ".join(f"{k} {len(v)}" for k, v in splits.items())
    md = render_markdown(
        reports,
        title="Phase 1 baselines",
        notes=[f"Data: {args.data} ({sizes} sentences, split by title)."],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    return 0


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
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "baselines":
        return _baselines(args)
    return 1  # pragma: no cover - argparse enforces a known command


if __name__ == "__main__":
    sys.exit(main())
