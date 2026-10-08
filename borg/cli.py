"""CLI entry point for borg-cube.

Commands
--------
borg train --component <component> <train_file> <dev_file> <model_file>
borg test <model_file> <input_file> <output_file>
"""
from __future__ import annotations

import argparse
import sys

from src.config import BorgConfig
from src.pipeline.pipeline import BorgPipeline


def _build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="borg",
        description="borg-cube NLP pipeline CLI",
    )
    sub = top.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = sub.add_parser("train", help="Train a pipeline component")
    train_p.add_argument(
        "--component",
        choices=["tokenizer", "tagger", "parser", "lemmatizer"],
        required=True,
        help="Component to train",
    )
    train_p.add_argument("train_file", help="Path to CoNLL-U training file")
    train_p.add_argument("dev_file", help="Path to CoNLL-U development file")
    train_p.add_argument("model_file", help="Model folder; the component is saved in its subfolder")
    train_p.add_argument("--resume", action="store_true", help="Resume from the component's last checkpoint")

    # Optional training hyper-parameter overrides
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--device", type=str, default=None)
    train_p.add_argument("--lang", type=str, default="en")

    # ---- test ----
    test_p = sub.add_parser("test", help="Run inference with a trained model/pipeline")
    test_p.add_argument("model_file", help="Model directory (or pipeline root)")
    test_p.add_argument("input_file", help="Input file (.txt for raw text, .conllu for annotated)")
    test_p.add_argument("output_file", help="Output CoNLL-U file")
    test_p.add_argument(
        "--component",
        choices=["tokenizer", "tagger", "parser", "lemmatizer"],
        default=None,
        help="Single component to apply (default: auto-detect from directory contents)",
    )
    test_p.add_argument("--device", type=str, default=None)
    test_p.add_argument("--lang", type=str, default="en")

    return top


def _make_config(args: argparse.Namespace) -> BorgConfig:
    cfg = BorgConfig(lang=getattr(args, "lang", "en"))
    if getattr(args, "epochs", None) is not None:
        cfg.num_epochs = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "lr", None) is not None:
        cfg.learning_rate = args.lr
    if getattr(args, "device", None) is not None:
        cfg.device = args.device
    return cfg


def cmd_train(args: argparse.Namespace) -> None:
    config = _make_config(args)
    pipeline = BorgPipeline(config)
    pipeline.train_component(
        args.component, args.train_file, args.dev_file, args.model_file,
        resume=args.resume,
    )
    print(
        f"Saved {args.component} last and best models to "
        f"{args.model_file}/{args.component}/{{last,best}}"
    )


def cmd_test(args: argparse.Namespace) -> None:
    import os

    config = _make_config(args)
    pipeline = BorgPipeline(config)

    if args.component is not None:
        # Single component test
        pipeline.load_component(args.component, args.model_file)
    else:
        # Auto-detect: try to load whichever component directories exist
        for comp in ["tokenizer", "tagger", "parser", "lemmatizer"]:
            comp_dir = os.path.join(args.model_file, comp)
            if os.path.isdir(comp_dir):
                pipeline.load_component(comp, comp_dir)
            elif os.path.isfile(os.path.join(args.model_file, "borg_config.json")):
                # model_file IS a single component directory
                # figure out which one
                import json
                with open(os.path.join(args.model_file, "borg_config.json")) as f:
                    info = json.load(f)
                detected = info.get("component")
                if detected:
                    pipeline.load_component(detected, args.model_file)
                break

    pipeline.test(args.model_file, args.input_file, args.output_file)


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
