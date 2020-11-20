from __future__ import annotations

import os

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal

BASE_FOLDER = "Data/"


def load_ascii_points(filepath: str) -> pd.DataFrame:
    """
    Load an ASCII ply point cloud into a Pandas DataFrame.

    :param filepath: The path to the point cloud.

    :returns: A DataFrame with fixed float64 dtypes and correct column names.
    """
    columns: list[str] = []
    header_stop = 0

    # Read the header in the ASCII formatted file to find the column names and where the header stops
    with open(filepath) as infile:
        for i, line in enumerate(infile):
            # If it's a column description
            if line.startswith("property"):
                # Format: "property dtype column_name"
                _, _, name = line.split(" ", maxsplit=3)

                # Append the column without newline characters and remove the "scalar_" prefixes
                columns.append(name.replace("\n", "").replace("scalar_", ""))

            # Stop if it reached the end of the header.
            if "end_header" in line:
                header_stop = i + 1
                break

    # Read the file as a space delimited file, assume that the dtype is float64, and skip the header
    data = pd.read_csv(filepath, sep=" ", header=None, names=columns,
                       dtype=np.float64, skiprows=header_stop, index_col=False)

    data.name = os.path.splitext(os.path.basename(filepath))[0]
    return data


def load_points(filepath: str) -> pd.DataFrame:
    """
    Load a binary point cloud into a Pandas DataFrame.

    :param filepath: The path to the point cloud to load.
    :returns: The loaded points.
    """
    pipeline_template = jinja2.Template("""
    [
        "{{ filepath }}"
    ]
    """)

    pipeline = pdal.Pipeline(pipeline_template.render({"filepath": filepath}))
    pipeline.execute()

    # Read the first array (the only one) and convert it to a dataframe
    data = pd.DataFrame(pipeline.arrays[0])
    # Set the name of the dataframe to the filename without an extension
    data.name = os.path.splitext(os.path.basename(filepath))[0]

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


def preprocess_points(points: pd.DataFrame, glacier_threshold: float = 70.0) -> pd.DataFrame:
    """
    Run the main analysis.
    """
    name = points.name

    points["gray"] = points[["red", "green", "blue"]].mean(axis=1)

    bin_number = 100
    gray_bins = np.linspace(0, 255, num=bin_number)
    histogram, _ = np.histogram(points["gray"], bins=gray_bins)
    most_common_value = float(np.argwhere(histogram == histogram.max()) * (255 / bin_number))

    for dimension in ["x", "y", "z"]:
        points[f"{dimension}diff"] = points[f"n{dimension}"] * points["M3C2_distance"]

    points.name = name
    points.drop(points[points["gray"] > most_common_value + glacier_threshold].index, inplace=True)
    return points


def plot(points: pd.DataFrame) -> None:
    """
    Plot the results.

    """
    plt.subplot(211)

    hist = np.histogram(points["gray"], bins=100)[0]
    max_index = np.argwhere(hist == hist.max())

    plt.bar(np.linspace(0, 255, num=hist.shape[0]), hist, width=2.6)

    current_ylim = plt.gca().get_ylim()
    plt.vlines(x=max_index * (255 / hist.shape[0]), ymin=0, ymax=current_ylim[1], color="black")
    plt.vlines(x=max_index * (255 / hist.shape[0]) + 80, ymin=0, ymax=current_ylim[1], color="red")

    plt.subplot(212)

    median = np.nanmedian(points["M3C2_distance"])
    plt.hist(points["M3C2_distance"], bins=200)

    plt.vlines(x=median, ymin=0, ymax=plt.gca().get_ylim()[1], color="black")
    plt.show()


if __name__ == "__main__":
    all_tiepoints = load_all_point_clouds()
    for tiepoints in all_tiepoints:
        preprocess_points(tiepoints)

    plt.figure(figsize=(16, 10), dpi=150)

    bins = np.linspace(-7, 7, num=150)
    colors = ["lightgreen", "orange", "skyblue"]
    for i, tiepoints in enumerate(all_tiepoints):

        for j, col in enumerate(["xdiff", "ydiff", "zdiff"]):
            if j == 2:
                plt.text(x=0.5, y=0.9, s=tiepoints.name, fontsize=8,
                         transform=plt.gca().transAxes, ha="center", va="center")
            plt.subplot(len(all_tiepoints), 3, (i * 3) + j + 1)

            plt.hist(tiepoints[col], bins=bins, color=colors[j])

            if i != len(all_tiepoints) - 1:
                plt.gca().set_xticklabels([])
            else:
                plt.xlabel(col)

            plt.grid()

    plt.tight_layout(h_pad=0)
    plt.show()
