"""Command line interface.

asag-eval evaluate predictions.jsonl --out results
asag-eval compare model.jsonl baseline.jsonl --out results
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from asag_eval import __version__
from asag_eval.data import load_predictions
from asag_eval.report import compare, evaluate, render_comparison, render_evaluation, write_json


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asag-eval",
        description="Reproducible evaluation of automated short-answer grading models.",
    )
    parser.add_argument("--version", action="version", version=f"asag-eval {__version__}")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    shared.add_argument(
        "--n-boot", type=_int_at_least(1), default=2000, help="bootstrap resamples (default: 2000)"
    )
    shared.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    shared.add_argument(
        "--n-classes", type=_int_at_least(2), default=3, help="scores are 0..N-1 (default: 3)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("evaluate", parents=[shared], help="evaluate one model")
    one.add_argument("predictions", type=Path, help="JSONL file in the predictions schema")
    one.add_argument(
        "--bins", type=_int_at_least(1), default=10, help="confidence bins for ECE (default: 10)"
    )
    one.add_argument("--no-plots", action="store_true", help="write tables only, no figures")

    two = commands.add_parser("compare", parents=[shared], help="paired comparison A - B")
    two.add_argument("a", type=Path, help="predictions of model A")
    two.add_argument("b", type=Path, help="predictions of model B on the same answers")
    return parser


def _evaluate(args: argparse.Namespace) -> str:
    preds = load_predictions(args.predictions, args.n_classes)
    result = evaluate(preds, args.n_boot, args.seed, args.bins)
    args.out.mkdir(parents=True, exist_ok=True)
    figures: tuple[str, ...] = ()
    if not args.no_plots:
        # Imported here so that tables can be produced without loading matplotlib.
        from asag_eval.plots import write_figures

        figures = write_figures(result, preds, args.out)
    write_json(result, args.out / "metrics.json")
    report = render_evaluation(result, figures)
    (args.out / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return report


def _compare(args: argparse.Namespace) -> str:
    result = compare(
        load_predictions(args.a, args.n_classes),
        load_predictions(args.b, args.n_classes),
        args.n_boot,
        args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(result, args.out / "comparison.json")
    report = render_comparison(result)
    (args.out / "comparison.md").write_text(report, encoding="utf-8", newline="\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = _evaluate(args) if args.command == "evaluate" else _compare(args)
    except (OSError, ValueError) as exc:
        # Bad input is the user's to fix: one clear line and exit code 2, not a traceback.
        parser.exit(2, f"asag-eval: error: {exc}\n")
    # A file name the console cannot encode must not fail a run whose files are written.
    sys.stdout.reconfigure(errors="replace")
    # Figures are files, not terminal output: print the tables only.
    print("\n".join(line for line in report.splitlines() if not line.startswith("![")).rstrip())
    print(f"Written to {args.out}")
