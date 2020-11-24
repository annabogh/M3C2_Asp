from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_ascii_points(filepath: str) -> pd.DataFrame:
    """
    Load an ASCII ply point cloud into a Pandas DataFrame.

    :param filepath: The path to the point cloud.

    :returns: A DataFrame with fixed float64 dtypes and correct column names.
    """
    column_names: list[str] = []
    header_stop = 0

    # Read the header in the ASCII formatted file to find the column names and where the header stops
    with open(filepath) as infile:
        for i, line in enumerate(infile):
            # If it's a column description
            if line.startswith("property"):
                # Format: "property dtype column_name"
                _, _, name = line.split(" ", maxsplit=3)

                # Append the column without newline characters and remove the "scalar_" prefixes
                column_names.append(name.replace("\n", "").replace("scalar_", ""))

            # Stop if it reached the end of the header.
            if "end_header" in line:
                header_stop = i + 1
                break

    # Read the file as a space delimited file, assume that the dtype is float64, and skip the header
    data = pd.read_csv(filepath, sep=" ", header=None, names=column_names,
                       dtype=np.float64, skiprows=header_stop, index_col=False)

    data.name = os.path.splitext(os.path.basename(filepath))[0]\
        .replace("Points_", "")\
        .replace("_M3C2", "")\
        .replace("Hoegsnyta", "Høgsnyta")
    return data


def to_latex_table(medians: pd.DataFrame, errors: pd.DataFrame, decimals: int = 3) -> str:
    """
    Convert a DataFrame of medians and one of errors to a latex table in the form of median+-error.

    :param medians: A DataFrame of median values.
    :param errors: A DataFrame of error values (presumably standard deviations).
    :param decimals: How many decimals to round the values in the table with.

    :returns: A string containing a latex table.
    """
    # Begin the table. The horizontal alignment should be left (l) for the index and centered (c) for the data.
    lines: list[str] = [
        r"\begin{tabular}{l" + "c" * len(medians.columns) + "}",
        r"\toprule"
    ]

    # Broadcast the medians and errors to a new cell DataFrame of the format: $'median'\pm'error'$
    cells = "$" + medians.round(decimals).astype(str) + r"\pm" + errors.round(decimals).astype(str) + "$"
    # Make sure the index is sorted
    cells.sort_index(inplace=True)

    # Append a column names row in bold
    lines.append(" & ".join(
        [r"\textbf{" + column + "}" for column in [medians.index.name] + list(medians.columns)]) + r"\\")
    # Add a line
    lines.append(r"\midrule")

    # Loop through every row and append it as a latex-row
    for index, row in cells.iterrows():
        line = index + " & " + " & ".join(row) + r"\\"
        lines.append(line)

    # Finish the table
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)
