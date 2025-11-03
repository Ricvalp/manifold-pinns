"""Command line helpers for running the three-step pipeline."""

import argparse
import ast
from typing import Any, Dict

from . import generate_dataset, run_pinn_experiment, train_autoencoder
from .uae_eval import evaluate_autoencoder


def _parse_overrides(values: str) -> Dict[str, Any]:
    """Parse dot-separated key=value pairs into a nested dictionary."""
    overrides: Dict[str, Any] = {}
    for item in values.split(","):
        key, raw_value = item.split("=", maxsplit=1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value

        parts = key.strip().split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides


def main(argv=None) -> None:
    """CLI entry point for driving the full workflow."""
    parser = argparse.ArgumentParser(description="Run manifold PINN pipeline steps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Generate UAE datasets.")
    dataset_parser.add_argument("name", choices=["bunny", "coil", "square"])
    dataset_parser.add_argument(
        "--override",
        type=str,
        help="Comma separated key=value pairs for config overrides.",
    )

    uae_parser = subparsers.add_parser("uae", help="Train the universal autoencoder.")
    uae_parser.add_argument("name", choices=["bunny", "coil", "square"])
    uae_parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Regenerate the dataset before training.",
    )
    uae_parser.add_argument(
        "--override",
        type=str,
        help="Comma separated key=value pairs for config overrides.",
    )

    pinn_parser = subparsers.add_parser("pinn", help="Run PINN experiments.")
    pinn_parser.add_argument("experiment", choices=["eikonal", "wave", "diffusion"])
    pinn_parser.add_argument("config", help="Configuration module name.")
    pinn_parser.add_argument(
        "--mode",
        choices=["train", "eval", "generate_data"],
        default="train",
        help="Execution mode for the PINN experiment.",
    )
    pinn_parser.add_argument(
        "--override",
        type=str,
        help="Comma separated key=value pairs for config overrides.",
    )

    uae_eval_parser = subparsers.add_parser(
        "uae-eval", help="Evaluate a UAE checkpoint on the validation split."
    )
    uae_eval_parser.add_argument("name", choices=["bunny", "coil", "square"])
    uae_eval_parser.add_argument("run_id", help="Run identifier (checkpoint folder).")
    uae_eval_parser.add_argument(
        "--step",
        type=int,
        help="Optional checkpoint step to restore. Defaults to the latest.",
    )
    uae_eval_parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of validation samples to visualise.",
    )
    uae_eval_parser.add_argument(
        "--output",
        type=str,
        help="Optional output path for reconstruction plots.",
    )
    uae_eval_parser.add_argument(
        "--override",
        type=str,
        help="Comma separated key=value pairs for config overrides.",
    )

    args = parser.parse_args(argv)

    overrides = (
        _parse_overrides(args.override) if getattr(args, "override", None) else None
    )

    if args.command == "dataset":
        generate_dataset(args.name, overrides)
    elif args.command == "uae":
        train_autoencoder(args.name, overrides, create_dataset=args.create_dataset)
    elif args.command == "pinn":
        run_pinn_experiment(args.experiment, args.config, mode=args.mode, overrides=overrides)
    elif args.command == "uae-eval":
        evaluate_autoencoder(
            args.name,
            args.run_id,
            overrides=overrides,
            checkpoint_step=args.step,
            num_samples=args.samples,
            output_path=args.output,
        )
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
