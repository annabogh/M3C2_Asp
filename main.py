from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_FOLDER = "Data/"


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

    data.name = os.path.splitext(os.path.basename(filepath))[0].replace("Points_", "").replace("_M3C2", "")
    return data


def load_all_point_clouds(folder: str = BASE_FOLDER) -> list[pd.DataFrame]:
    """
    Load all ".ply" point clouds in a folder.

    :param folder: The folder to look in.
    :returns: A list of point clouds.
    """
    point_clouds: list[pd.DataFrame] = []
    for filename in os.listdir(folder):
        if not filename.endswith(".ply"):
            continue

        points = load_ascii_points(os.path.join(folder, filename))
        point_clouds.append(points)
    return point_clouds


def preprocess_points(points: pd.DataFrame, glacier_threshold: float = 50.0) -> pd.DataFrame:
    """
    Run the main analysis.
    """
    name = points.name

    points["gray"] = points[["red", "green", "blue"]].mean(axis=1)

    bin_number = 100
    gray_bins = np.linspace(0, 255, num=bin_number)
    histogram, _ = np.histogram(points["gray"], bins=gray_bins)
    # TODO: Fix below that there may be more than one maximum index
    most_common_value = np.argwhere(histogram == histogram.max())[0, 0] * (255 / bin_number)

    for dimension in ["x", "y", "z"]:
        points[f"{dimension}diff"] = points[f"n{dimension}"] * points["M3C2_distance"]

    cropped_points = points[points["gray"] < (most_common_value + glacier_threshold)]

    cropped_points.name = points.name

    return cropped_points


def plot(tiepoint_list: list[pd.DataFrame]) -> None:
    """
    Plot the results.

    """
    # TODO: Resize figure to fit an area smaller than an A4
    plt.figure(figsize=(16, 10), dpi=150)

    bins = np.linspace(-7, 7, num=150)
    colors = ["lightgreen", "orange", "skyblue"]
    for i, tiepoints in enumerate(tiepoint_list):

        for j, col in enumerate(["xdiff", "ydiff", "zdiff"]):
            plt.subplot(len(tiepoint_list), 3, (i * 3) + j + 1)

            # Make a title if in the middle column
            if j == 1:
                plt.text(x=0.5, y=0.9, s=tiepoints.name, fontsize=8,
                         transform=plt.gca().transAxes, ha="center", va="center")
            plt.hist(tiepoints[col], bins=bins, color=colors[j], density=True)

            # Remove x-tick labels for all except the last rows.
            if i != len(tiepoint_list) - 1:
                plt.gca().set_xticklabels([])
            # If on the last row, also add an x-label
            else:
                plt.xlabel(col)

            plt.grid()

    plt.tight_layout(h_pad=0)
    os.makedirs("Output", exist_ok=True)
    plt.savefig("Output/m3c2_error_histogram.png", dpi=600)
    plt.show()


def calc_statistics(point_list: list[pd.DataFrame]):

    column_names = ["xdiff", "ydiff", "zdiff", "M3C2_distance"]

    errors = pd.DataFrame(columns=column_names)

    for points in point_list:
        for column in column_names:
            median = points[column].median()
            std = points[column].std()
            error = f"{median:.3f}±{std:.3f}"
            errors.loc[points.name, column] = error

    errors.rename(columns={
        "xdiff": "Easting (Δm)",
        "ydiff": "Northing (Δm)",
        "zdiff": "Height (Δm)",
        "M3C2_distance": "Total error (Δm)"
    }, inplace=True)
    errors.index.name = "Locality"

    os.makedirs("Output", exist_ok=True)
    with open("Output/m3c2_error_table.tex", "w", encoding="utf-8") as outfile:
        out_text = errors.reset_index().to_latex(
            index=False,
            bold_rows=True,
            column_format="c" * (len(column_names) + 1))
        outfile.write(out_text)
    print(errors)


if __name__ == "__main__":
    all_tiepoints = load_all_point_clouds()
    preprocessed_tiepoints = []
    for tiepoints in all_tiepoints:
        preprocessed_tiepoints.append(preprocess_points(tiepoints))
    calc_statistics(all_tiepoints)

    plot(preprocessed_tiepoints)
