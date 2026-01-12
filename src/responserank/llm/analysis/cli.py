#!/usr/bin/env python3
"""
Analysis CLI for collecting experiment results and generating figures.

The workflow is to first collect data from wandb (which can be slow),
then generate plots from the cached data (which is fast and can be iterated).

Usage:
    python -m responserank.llm.analysis.cli collect results/llm \\
        --project-name responserank-llm --registry results/llm/registry.yaml

    python -m responserank.llm.analysis.cli generate results/llm paper/figures
"""

import argparse
import sys
from pathlib import Path

from responserank.llm.analysis.data_cache import collect_and_save_data, load_cached_data
from responserank.llm.analysis.dataset_analysis import generate_dataset_stats
from responserank.llm.analysis.experiments import load_experiments
from responserank.llm.analysis.paper_plots import generate_paper_plots


def main():
    parser = argparse.ArgumentParser(
        description="LLM RT Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Type of analysis to perform", required=True
    )

    collect_parser = subparsers.add_parser(
        "collect", help="Collect experiment data from wandb and save to cache files"
    )
    collect_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Output directory for cache files (will contain data.pkl and meta.json)",
    )
    collect_parser.add_argument(
        "--project-name", type=str, required=True, help="wandb project name"
    )
    collect_parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to experiment_registry.yaml",
    )
    collect_parser.add_argument(
        "--seed-filter", type=int, help="Filter results to specific seed (optional)"
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate all paper plots and tables from cached data",
    )
    generate_parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory containing cached data files (data.pkl and meta.json)",
    )
    generate_parser.add_argument(
        "figures_dir",
        type=Path,
        help="Directory where to save paper figures and tables",
    )

    args = parser.parse_args()

    if args.command == "collect":
        experiments = load_experiments(args.registry)

        egid_patterns = set()
        style_overrides = {}
        display_name_overrides = {}

        for exp in experiments.values():
            egid = exp["egid"]
            display_name_overrides[egid] = exp["display_name"]
            style_overrides[egid] = {
                "color": exp["color"],
                "marker": exp["marker"],
                "linestyle": exp["linestyle"],
            }
            egid_patterns.add(egid)

        collect_and_save_data(
            project_name=args.project_name,
            output_dir=args.cache_dir,
            egid_patterns=egid_patterns,
            experiments=experiments,
            style_overrides=style_overrides,
            display_name_overrides=display_name_overrides,
            seed_filter=args.seed_filter,
            command_line=" ".join(sys.argv),
        )

    elif args.command == "generate":
        df, metadata = load_cached_data(args.cache_dir)
        experiments = metadata["experiments"]
        generate_paper_plots(
            df, metadata, experiments=experiments, figures_dir=args.figures_dir
        )
        generate_dataset_stats(args.figures_dir / "dataset_stats.tex")


if __name__ == "__main__":
    main()
