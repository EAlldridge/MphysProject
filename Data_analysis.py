#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:35:19 2025

@author: matthewillsley
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scaling parameters for file 1 and file 2
len1 = 0.645
len2 = 0.3225

area1 = 0.416
area2 = 0.104
####


def import_files():
    # Define file paths (Assuming CSV files, convert Excel to CSV first)
    file1 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/Spreadsheets seperated/20x/20x_Nuclei.csv"
    file2 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/Spreadsheets seperated/40X/40x_Nuclei.csv"

    # image#,object#,area,eccentricity,form factor,perimeter,solidity
    columns_to_import = [0, 1, 2, 24, 28, 66, 67]

    scale_factors_file1 = np.array(
        [1, 1, area1, 1, 1, len1, 1])  # Example scaling factors
    scale_factors_file2 = np.array([1, 1, area2, 1, 1, len1, 1])

    # Import data using np.genfromtxt (Skip header row, adjust delimiter if needed)
    data1 = np.genfromtxt(file1, delimiter=",", skip_header=1,
                          usecols=columns_to_import)
    data2 = np.genfromtxt(file2, delimiter=",", skip_header=4,
                          usecols=columns_to_import)

    # Scale the data
    scaled_data1 = data1 * scale_factors_file1
    scaled_data2 = data2 * scale_factors_file2

    data_combined = np.vstack((scaled_data1, scaled_data2))

    print("import and combination completed")

    return data_combined, scaled_data1, scaled_data2


def histogram(data_combined, col_num, name):

    area_data = data_combined[:, col_num]
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(area_data, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + name)

    # Show the plot
    plt.show()
    plt.savefig(fname=name+".png", dpi=1000)


def overlay_histogram_area(data1, data2):

    area_data1 = data1[:, 2]
    area_data2 = data2[:, 2]

    bins = np.linspace(min(area_data1.min(), area_data2.min()),
                       max(area_data1.max(), area_data2.max()),
                       30)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(area_data1, bins, edgecolor='black', alpha=0.7, label="20X")
    plt.hist(area_data2, bins, edgecolor='black', alpha=0.7, label="40X")
    plt.xlabel("Object Area")
    plt.ylabel("Frequency")
    plt.title("Histogram of Object Areas")
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(fname="histogram_sep.png", dpi=1000)


data_combo, data1, data2 = import_files()


histogram(data_combo, 2, "Area")
histogram(data_combo, 3, "Eccentricity")
histogram(data_combo, 4, "Form factor")
histogram(data_combo, 5, "Perimeter")
histogram(data_combo, 6, "Solidity")

overlay_histogram_area(data1, data2)
