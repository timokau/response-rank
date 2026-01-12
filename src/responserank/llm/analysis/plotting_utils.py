"""Common plotting utilities for analysis scripts."""

import datetime
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def setup_plot_style():
    """Initialize consistent plot styling across all analysis scripts."""
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})


def get_loss_type_colors(loss_types: List[str]) -> Dict[str, str]:
    """Get consistent color mapping for loss types.

    Args:
        loss_types: List of loss type names

    Returns:
        Dictionary mapping loss type to color
    """
    palette = sns.color_palette("colorblind", n_colors=len(loss_types))
    return {lt: palette[i] for i, lt in enumerate(loss_types)}


def save_figure_reproducible(fig, output_path: str, **kwargs):
    """Save figure as PDF with fixed metadata for reproducible binary output.

    This is much more git friendly, only requiring commits when anything relevant actually changed.

    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the PDF
        **kwargs: Additional arguments passed to savefig
    """
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.05, "dpi": 300, **kwargs}

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, **save_kwargs)

        d = pdf.infodict()
        d["CreationDate"] = datetime.datetime(1990, 1, 1)
        d["ModDate"] = datetime.datetime(1990, 1, 1)
        d["Creator"] = "Matplotlib"
        d["Producer"] = "Matplotlib"


def warn_on_low_seed_counts(
    df: pd.DataFrame,
    *,
    experiments: Sequence[str],
    groupby_cols: Sequence[str],
    threshold: int,
    display_name_aliases: Dict[str, str],
) -> None:
    """Emit warnings when plotted points use fewer seeds than required.

    Args:
        df: Filtered dataframe backing the plot.
        experiments: Raw experiment display names included in the plot.
        groupby_cols: Columns describing the plotted grouping (e.g., fractions).
        threshold: Minimum allowed number of unique seeds.
        display_name_aliases: Mapping from raw display names to presentation aliases.

    Returns:
        None. Warnings are emitted to stderr.
    """

    def format_group_components(
        columns: Sequence[str], values: Sequence[object]
    ) -> str:
        pieces = []
        for column, value in zip(columns, values):
            if isinstance(value, float) and value.is_integer():
                value_repr = str(int(value))
            else:
                value_repr = f"{value}"
            pieces.append(f"{column}={value_repr}")

        return ", ".join(pieces)

    for experiment in experiments:
        subset = df[df["egid"] == experiment]

        legend_name = display_name_aliases.get(experiment, experiment)

        if len(groupby_cols) > 0:
            grouped_iter = subset.groupby(list(groupby_cols), dropna=False)
        else:
            grouped_iter = [((), subset)]

        for group_keys, group_df in grouped_iter:
            seed_count = int(group_df["seed"].dropna().nunique())

            if seed_count >= threshold:
                continue

            group_descriptor = format_group_components(groupby_cols, group_keys)
            suffix = f" @ {group_descriptor}" if group_descriptor else ""
            print(
                (
                    f"WARNING {legend_name}{suffix} has {seed_count} seeds (<{threshold})"
                ),
                file=sys.stderr,
            )
