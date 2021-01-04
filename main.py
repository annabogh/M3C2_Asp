from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utilities

BASE_FOLDER = "Data/"


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

        points = utilities.load_ascii_points(os.path.join(folder, filename))
        point_clouds.append(points)
    return point_clouds


def preprocess_points(points: pd.DataFrame, glacier_threshold: float = 50.0) -> pd.DataFrame:
    """
    Preprocess points by removing bright spots (snow/ice) and adding x/y/x-diff columns.

    :param points: The tie points to preprocess.
    :param glacier_threshold: The maximum allowed brightness value, counted from the most normal brightness value.

    :returns: A new DataFrame with the same shape with the new columns
    """
    new_points = points.copy()

    for dimension in ["x", "y", "z"]:
        new_points[f"{dimension}diff"] = new_points[f"n{dimension}"] * new_points["M3C2_distance"]

    new_points["gray"] = new_points[["red", "green", "blue"]].mean(axis=1)
    cropped_points = new_points[new_points["gray"] < (new_points["gray"].median() + glacier_threshold)]

    cropped_points.name = points.name

    return cropped_points


def plot(tiepoint_list: list[pd.DataFrame]) -> None:
    """
    Plot the distribution plots.

    :param tiepoint_list: A list of Pandas DataFrames representing the tie point clouds.
    """
    # TODO: Resize figure to fit an area smaller than an A4
    plt.figure(figsize=(16, 10), dpi=150)

    x_labels = {"xdiff": "Δ Easting (m)", "ydiff": "Δ Northing (m)", "zdiff": "Δ Height (m)"}

    magnitude = 7
    bins = np.linspace(-magnitude, magnitude, num=150)
    colors = ["lightgreen", "orange", "skyblue"]
    for i, tiepoints in enumerate(tiepoint_list):

        for j, col in enumerate(["xdiff", "ydiff", "zdiff"]):
            plt.subplot(len(tiepoint_list), 3, (i * 3) + j + 1)

            # Make a title if in the middle column
            if j == 1:
                plt.text(x=0.5, y=0.9, s=tiepoints.name, fontsize=8,
                         transform=plt.gca().transAxes, ha="center", va="center")
            plt.hist(tiepoints[col], bins=bins, color=colors[j], density=True)
            median = np.nanmedian(tiepoints[col])
            standard_dev = np.std(tiepoints[col])
            text = f"M = {median:.2f} m\nσ = {standard_dev:.2f} m"
            plt.text(x=0.75, y=0.9, s=text, fontsize=7, transform=plt.gca().transAxes, ha="left", va="top")

            # Remove x-tick labels for all except the last rows.
            if i != len(tiepoint_list) - 1:
                plt.gca().set_xticklabels([])
            # If on the last row, also add an x-label
            else:
                plt.xlabel(x_labels[col])

            plt.xlim(-magnitude, magnitude)

            plt.grid()

    plt.tight_layout(h_pad=0)
    os.makedirs("Output", exist_ok=True)
    plt.savefig("Output/m3c2_error_histogram.png", dpi=600)
    plt.show()


def calc_statistics(tiepoint_list: list[pd.DataFrame]):
    """
    Calculate the statistics of each dimension's error distribution.

    :param tiepoint_list: A list of Pandas DataFrames representing the tie point clouds.
    """
    # Choose which columns should be included in the statistics
    column_names = ["xdiff", "ydiff", "zdiff", "M3C2_distance"]

    # Create two identical (empty) DataFrames for the medians and standard deviations
    medians = pd.DataFrame(columns=column_names, dtype=np.float64)
    stds = medians.copy()

    # Iteratively fill the above dataframes with data
    for points in tiepoint_list:
        for column in column_names:
            median = points[column].median()
            std = points[column].std()

            # Add the above results to the appropriate DataFrames
            medians.loc[points.name, column] = median
            stds.loc[points.name, column] = std

    # Rename the columns and indices of the two DataFrames to look nicer
    for data in (medians, stds):
        data.rename(columns={
            "xdiff": "Easting ($\Delta$m)",
            "ydiff": "Northing ($\Delta$m)",
            "zdiff": "Height ($\Delta$m)",
            "M3C2_distance": "Total error ($\Delta$m)"
        }, inplace=True)
        data.index.name = "Locality"

    # Make the LaTeX table from the medians and stds
    latex_table = utilities.to_latex_table(medians=medians, errors=stds)

    # Write it to an output file
    os.makedirs("Output", exist_ok=True)
    with open("Output/m3c2_error_table.tex", "w", encoding="utf-8") as outfile:
        outfile.write(latex_table)

    print(latex_table)


if __name__ == "__main__":
    # Load all the tie point clouds
    all_tiepoints = load_all_point_clouds()
    print("Loaded all tie points")
    # Preprocess all the tie point clouds.
    preprocessed_tiepoints = [preprocess_points(tiepoints, glacier_threshold=70) for tiepoints in all_tiepoints]
    print("Preprocessed all tie points")
    # Calculate statistics on them.
    calc_statistics(preprocessed_tiepoints)
    print("Calculated statistics")
    print("Plotting...")
    # Plot their error distribution
    plot(preprocessed_tiepoints)
