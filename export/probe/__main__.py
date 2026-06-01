from __future__ import annotations
import argparse
import sys

from .config import Config
from .runner import run_probes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run behavioral probes against a fused model.",
        epilog=(
            "Thresholds load from .env / env vars (PROBE_THRESHOLD_<DOMAIN>) "
            "then CLI overrides. "
            "Example: PROBE_THRESHOLD_CALENDAR=0.85 or --threshold CALENDAR=0.85"
        ),
    )
    parser.add_argument(
        "--model", required=True, help="Path to fused model (model/merged)"
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", default=None)
    parser.add_argument(
        "--threshold",
        metavar="DOMAIN=RATE",
        action="append",
        default=[],
        help="Override a domain threshold. Repeatable. Takes precedence over env vars.",
    )
    args = parser.parse_args()

    cli_thresholds: dict[str, float] = {}
    for spec in args.threshold:
        try:
            domain, rate = spec.split("=", 1)
            cli_thresholds[domain.upper()] = float(rate)
        except ValueError:
            print(
                f"Invalid --threshold format: {spec!r}. Use DOMAIN=RATE.",
                file=sys.stderr,
            )
            sys.exit(1)

    config = Config.from_env(model_path=args.model).apply_cli_overrides(
        max_tokens=args.max_tokens,
        verbose=args.verbose if args.verbose else None,
        thresholds=cli_thresholds,
    )
    run_probes(config)


if __name__ == "__main__":
    main()
