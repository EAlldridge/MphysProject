#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:54:18 2025

@author: matthewillsley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:20:27 2025

@author: matthewillsley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:35:19 2025

@author: matthewillsley
"""

# Scaling parameters for file 1 and file 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import curve_fit
len1 = 0.645  # in micron 20x
len2 = 0.3225  # 40x


d_lim_1 = 0.573

area1 = 0.416  # in micron^2
area2 = 0.104
d_lim_2 = 0.382
k = 0.5
####


def import_files():
    # Define file paths (Assuming CSV files, convert Excel to CSV first)
    # file1 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/Spreadsheets seperated/20x/20x_Nuclei.csv"
    # file2 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/Spreadsheets seperated/40X/40x_Nuclei.csv"
    file3 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/20x_2Nuclei.csv"
    file4 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/40x_2Nuclei.csv"

    file5 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/40x_segmentation_1.1xNuclei.csv"
    file6 = "/Users/matthewillsley/Documents/UOM/Year 4/Project sem 2 /Cell profiler/40x_segmentation_0.9xNuclei.csv"
    # image#,object#,area,eccentricity,form factor,perimeter,solidity

    columns_to_import = [0, 1, 2, 24, 28, 66, 67]

    scale_factors_file1 = np.array(
        [1, 1, area1, 1, 1, len1, 1])  # Example scaling factors
    scale_factors_file2 = np.array([1, 1, area2, 1, 1, len2, 1])

    # Import data using np.genfromtxt (Skip header row, adjust delimiter if needed)
    # data1 = np.genfromtxt(file1, delimiter=",", skip_header=1,
    # usecols=columns_to_import)
    # data2 = np.genfromtxt(file2, delimiter=",", skip_header=4,
    # usecols=columns_to_import)
    data3 = np.genfromtxt(file3, delimiter=",", skip_header=1,
                          usecols=columns_to_import)
    data4 = np.genfromtxt(file4, delimiter=",", skip_header=1,
                          usecols=columns_to_import)

    data_harsh = np.genfromtxt(file5, delimiter=",", skip_header=1,
                               usecols=columns_to_import)
    data_easy = np.genfromtxt(file6, delimiter=",", skip_header=1,
                              usecols=columns_to_import)

    # Scale the data
    # scaled_data1 = data1 * scale_factors_file1
    # scaled_data2 = data2 * scale_factors_file2
    scaled_data3 = data3 * scale_factors_file1
    scaled_data4 = data4 * scale_factors_file2

    scaled_harsh = data_harsh * scale_factors_file2
    scaled_easy = data_easy * scale_factors_file2

    # data_combined = np.vstack((scaled_data1, scaled_data2))

    pixel_err_20x = scaled_data3[:, 5] * len1

    pixel_err_perim_20x = np.sqrt(scaled_data3[:, 5])*len1

    diff_lim_20x = (np.pi*d_lim_1**2)/4

    diff_lim_20x_perim = np.pi*d_lim_1

    pixel_err_40x = scaled_data4[:, 5] * len2

    pixel_err_perim_40x = np.sqrt(scaled_data4[:, 5]) * len2

    diff_lim_40x = (np.pi*d_lim_2**2)/4

    diff_lim_40x_perim = np.pi*d_lim_2

    pixel_err_20x = pixel_err_20x.reshape(-1, 1)
    pixel_err_40x = pixel_err_40x.reshape(-1, 1)

    pixel_err_perim_20x = pixel_err_perim_20x.reshape(-1, 1)
    pixel_err_perim_40x = pixel_err_perim_40x.reshape(-1, 1)

    diff_err_20x_col = np.full((scaled_data3.shape[0], 1), diff_lim_20x)
    diff_err_40x_col = np.full((scaled_data4.shape[0], 1), diff_lim_40x)

    diff_err_20x_perim_col = np.full(
        (scaled_data3.shape[0], 1), diff_lim_20x_perim)
    diff_err_40x_perim_col = np.full(
        (scaled_data4.shape[0], 1), diff_lim_40x_perim)

    scaled_data3 = np.hstack((scaled_data3, pixel_err_20x,
                             diff_err_20x_col, pixel_err_perim_20x, diff_err_20x_perim_col))
    scaled_data4 = np.hstack((scaled_data4, pixel_err_40x,
                             diff_err_40x_col, pixel_err_perim_40x, diff_err_40x_perim_col))

    data_combined_new = np.vstack((scaled_data3, scaled_data4))

    print("import and combination completed")

    return data_combined_new, scaled_data3, scaled_data4, scaled_harsh, scaled_easy


def histogram(data_combined, col_num, name):

    area_data = data_combined[:, col_num]
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(area_data, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + name)

    # Show the plot
    plt.show()
    plt.savefig(fname=name+".png", dpi=1000)


def overlay_histogram_area(data1, data2, title):

    area_data1 = data1[:, 2]
    area_data2 = data2[:, 2]

    bins = np.linspace(min(area_data1.min(), area_data2.min()),
                       max(area_data1.max(), area_data2.max()),
                       30)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(area_data1, bins, edgecolor='black', alpha=0.7, label="OLD")
    plt.hist(area_data2, bins, edgecolor='black', alpha=0.7, label="NEW")

    plt.xlabel("Object Area")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(fname="histogram_sep_comparison.png", dpi=1000)


def nuclear_area_factor(area, form_factor):
    return area/form_factor


def circum_div_form_factor(circum, form_factor):
    return circum/form_factor


def log_norm_peak_scaled(x, A, mu, sigma):
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    # Find peak of the PDF for these parameters
    mode = np.exp(mu - sigma**2)
    peak_height = lognorm.pdf(mode, s=sigma, scale=np.exp(mu))
    # Scale PDF so its peak is A
    return A * (pdf / peak_height)


def fitting(data, data_errors=None, bins=25, n_resamples=1, title="title"):
    """
    Fits a histogram of object areas (with optional measurement errors) to a log-normal distribution.

    Parameters:
    - data: array of measured areas
    - data_errors: array of uncertainties on each area
    - bins: number of bins for histogram
    - n_resamples: number of resamples per data point (default 1 = simple histogram, higher = smoothed histogram)
    """

    if data_errors is not None:
        # Resample data according to their uncertainties
        resampled_data = []
        for d, err in zip(data, data_errors):
            # model the data points as normal functions
            samples = np.random.normal(d, err, n_resamples)
            resampled_data.extend(samples)
        resampled_data = np.array(resampled_data)
    else:
        resampled_data = data

    # Build histogram

    counts, bin_edges = np.histogram(resampled_data, bins=bins)
    # use the resampled data for the histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # RENORMALIZE histogram to original number of objects
    # This is to remove the extra events created by the resampling
    n_objects = len(data)
    counts_normalized = counts * (n_objects / np.sum(counts))

    # Corrected Poisson errors (after normalization)

    errors_normalized = np.sqrt(counts_normalized)

    # Initial guess based on log-transformed resampled data
    positive_resampled = resampled_data[resampled_data > 0]

    # Initial guess of the fit parameters using standard normal and log
    initial_guess = [
        max(counts_normalized),  # approximate peak height
        np.mean(np.log(positive_resampled)),
        np.std(np.log(positive_resampled))
    ]

    # Fitting using the log_normal and the bin centres found previously.
    params, cov = curve_fit(
        log_norm_peak_scaled, bin_centers, counts_normalized,
        p0=initial_guess, sigma=errors_normalized,
        absolute_sigma=True
    )

    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 500)
    y_fit = log_norm_peak_scaled(x_fit, *params)

    perr = np.sqrt(np.diag(cov))

    corr_param, corr_err = convert_params(params, perr)

    model_counts = log_norm_peak_scaled(bin_centers, *params)

    chi_squared = np.sum(
        ((counts_normalized - model_counts) / errors_normalized) ** 2)

    dof = len(counts_normalized) - len(params)
    reduced_chi_squared = chi_squared / dof

    print("The reduced chi squared is", reduced_chi_squared)
    print(f"Fitted parameters:")
    print(f"A     = {params[0]:.2f} ± {perr[0]:.2f}")
    print(f"mu    = {corr_param[0]:.2f} ± {corr_err[0]:.4f}")
    print(f"sigma = {corr_param[1]:.2f} ± {corr_err[1]:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(bin_centers, counts_normalized, yerr=errors_normalized,
                 fmt='o', capsize=3, label='Data', color='red')
    hist_normalized = counts * (n_objects / np.sum(counts))

    # Plot histogram using normalized heights
    plt.bar(bin_centers, hist_normalized, width=(bin_edges[1] - bin_edges[0]),
            edgecolor='black', alpha=0.3, color='grey', label="Histogram")
    plt.plot(x_fit, y_fit, '-',
             label=f'Log-normal Fit', color='orange')

    plt.xlabel(title)
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()
    plt.savefig(fname='Nuclear Area Fit.png', dpi=1000)
    plt.show()


def area_vs_circ(data_combo):
    '''
    Plots the relationship between object area and circularity (form factor),
    including error bars.
    '''

    # Sort by area
    data_combo = data_combo[data_combo[:, 2].argsort()]

    area = data_combo[:, 2]
    circ = data_combo[:, 4]

    # Assuming error columns:
    area_error = data_combo[:, 13]   # Error in area
    circ_error = data_combo[:, 15]   # Error in circularity

    plt.figure(figsize=(8, 6))
    plt.errorbar(area, circ, xerr=area_error, yerr=circ_error,
                 fmt='o', ecolor='gray', capsize=3, alpha=0.8, label='Data with errors')

    plt.xlabel(r"Object Area ($\mu\mathrm{m}^2$)")
    plt.ylabel("Form Factor")
    plt.title("Circularity vs. Area")
    plt.legend()
    plt.tight_layout()
    plt.show()


def convert_params(params, perr):

    mean = np.exp(params[1]+(params[2]**2)/2)
    var_mean = (mean**2) * (perr[1]**2 + (params[2]*perr[2])**2)
    err_mean = np.sqrt(var_mean)

    var = (np.exp(params[2]**2)-1)*(np.exp((2*params[1]) + params[2]**2))
    std = np.sqrt(var)

    brkt_term = (1 + (np.exp(params[2]**2)/(np.exp(params[2]**2)-1)))**2

    var_std = var * (perr[1]**2 + (params[2]**2 * brkt_term * perr[2]**2))

    err_std = np.sqrt(var_std)

    corr_param = mean, std
    corr_err = err_mean, err_std

    return corr_param, corr_err


def simple_hist(data, title):

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, edgecolor='black',
             alpha=0.7)

    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()
    plt.savefig(fname=title+".png", dpi=1000)


def error_finding(benchmark, harsh, easy):

    A_benchmark = np.sum(benchmark[:, 2]) / len(benchmark[:, 2])

    A_harsher = np.sum(harsh[:, 2]) / len(harsh[:, 2])

    A_easier = np.sum(easy[:, 2]) / len(easy[:, 2])

    delta_A = np.sqrt(((A_harsher - A_benchmark)**2 +
                      (A_easier - A_benchmark)**2) / 2)

    print("error in the Area is: ", delta_A)
    perc_err_A = (delta_A/A_benchmark)*100
    print("% error in Area is: ", perc_err_A)

    P_benchmark = np.sum(benchmark[:, 5]) / len(benchmark[:, 5])

    P_harsher = np.sum(harsh[:, 5]) / len(harsh[:, 5])

    P_easier = np.sum(easy[:, 5]) / len(easy[:, 5])

    delta_P = np.sqrt(((P_harsher - P_benchmark)**2 +
                      (P_easier - P_benchmark)**2) / 2)

    print("error in the Perimeter is: ", delta_P)

    perc_err_P = (delta_P/P_benchmark)*100
    print("% error in Perimeter is: ", perc_err_P)

    return perc_err_A, perc_err_P


def combine_area_error(data_combo):

    pixel_err = data_combo[:, 7]
    diff_err = data_combo[:, 8]
    seg_area_err = data_combo[:, 11]

    combo_err = np.sqrt(pixel_err**2 + seg_area_err**2 + diff_err**2)

    combo_err = combo_err.reshape(-1, 1)

    data_combo = np.hstack((data_combo, combo_err))

    return data_combo


def combine_perimeter_error(data_combo):

    seg_perim_err = data_combo[:, 12]
    diff_perim_error = data_combo[:, 10]
    pixel_perim_error = data_combo[:, 9]

    combo_perim_err = np.sqrt(
        seg_perim_err**2 + diff_perim_error**2 + pixel_perim_error**2)
    combo_perim_err = combo_perim_err.reshape(-1, 1)
    data_combo = np.hstack((data_combo, combo_perim_err))

    return data_combo


def circ_err(data):

    perim = data[:, 5]
    area = data[:, 2]
    perim_err = data[:, 14]
    area_err = data[:, 13]

    dC_dA = (4 * np.pi) / (perim ** 2)
    dC_dP = (-8 * np.pi * area) / (perim ** 3)

    circ_var = (dC_dA * area_err) ** 2 + (dC_dP * perim_err) ** 2
    circ_err = np.sqrt(circ_var)

    circ_err = circ_err.reshape(-1, 1)
    data = np.hstack((data, circ_err))

    return data


data_combo, data1, data2, data_harsh, data_easy = import_files()


perc_error_area, perc_error_perim = error_finding(data2, data_harsh, data_easy)

area_errors = data_combo[:, 2] * (perc_error_area / 100.0)

# Reshape area_errors to be a column vector
area_errors = area_errors.reshape(-1, 1)

data_combo = np.hstack((data_combo, area_errors))


perim_errors = data_combo[:, 5] * (perc_error_perim / 100.0)

# Reshape area_errors to be a column vector
perim_errors = perim_errors.reshape(-1, 1)

data_combo = np.hstack((data_combo, perim_errors))

data_combo = combine_area_error(data_combo)
data_combo = combine_perimeter_error(data_combo)


data_combo = circ_err(data_combo)

histogram(data_combo, 2, "Area")

'''


histogram(data_combo, 3, "Eccentricity")
histogram(data_combo, 4, "Form factor")
histogram(data_combo, 5, "Perimeter")
histogram(data_combo, 6, "Solidity")'''

fitting(data_combo[:, 2], data_combo[:, -3], bins=30,
        n_resamples=1000, title=r"Nuclear area ($\mu\mathrm{m}^2$)")
fitting(data_combo[:, 5], data_combo[:, -2], bins=30,
        n_resamples=1000, title=r"Nuclear Perimeter ($\mu\mathrm{m}$)")
area_vs_circ(data_combo)

simple_hist(nuclear_area_factor(data_combo[:, 2], data_combo[:, 4]), "NAF")
simple_hist(circum_div_form_factor(
    data_combo[:, 5], data_combo[:, 4]), "cirum div form factor")

'''overlay_histogram_area(data1, data3, "20X comparison")
overlay_histogram_area(data2, data4, "40X comparison")
overlay_histogram_area(data_combo, data_combo_new, "Overall comparison")'''
