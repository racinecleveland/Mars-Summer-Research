"""
Some global settings
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from glob import glob

from muldoon.read_data import read_Perseverance_PS_data, make_seconds_since_midnight
from muldoon.utils import break_at_gaps, write_out_plot_data

# read_Perseverance_MEDA_data = read_Perseverance_PS_data

# Boise State official colors in hex
# boisestate.edu/communicationsandmarketing/brand-standards/colors/
BoiseState_blue = "#0033A0"
BoiseState_orange = "#D64309"
aspect_ratio = 16./9

window_size = 500./3600 # window size is 500 seconds
matched_filter_depth = 1./np.pi
detection_threshold = 7.

# Find vortices - fwhm_factor = 4 keeps the time-series grabbed for each vortex from sprawling.
fwhm_factor=3

# Distance between peaks in convolution - this one needs to be a little larger than usual.
distance_between_peaks = 30

# Directory where data are stored
data_dir = "../data/data_derived_env/"

# Sols with dataset issues:
sols_to_exclude = ["0001", "0002", "0004", "0009", "0014", "0019", "0022"]
def exclude_sols(file_list, sols_to_exclude=sols_to_exclude):
    """
    Drop sols with problematic datasets from file list

    Args:
        file_list (list of str): list of file names
        sol_to_exclude (list of str, optional): list of sols to drop

    Returns:
        file list with sols excluded (list of str)
    """
    copy_file_list = file_list.copy()

    for i in range(len(file_list)):
        for j in range(len(sols_to_exclude)):
            if(file_list[i].find(sols_to_exclude[j]) != -1):
                copy_file_list.remove(file_list[i])

    return copy_file_list

def culled_vortices(vortex_fit_params_file=\
        "../data/data_derived_env/fit_all_vortices.csv"):
    """
    Some vortex signals are spurious. This package reads in the data file
    that contains best-fit parameters for all the vortices and then filters out
    the ones that don't satisy certain criteria.

    Args:
        vortex_fit_params_file (str, optional): path to data file

    Returns:
        data from data file
    """

    Gamma_max = 250. # max duration in seconds
    DeltaP_SNR = 5. # minimum signal-to-noise ratio

    data = np.genfromtxt(vortex_fit_params_file, names=True, delimiter=',')
    ind = (data['Gamma'] < Gamma_max) &\
            (data['DeltaP']/data['DeltaP_unc'] > DeltaP_SNR)

    return data[ind]

def scatterplot_with_cum_hist(x, y, xerr=None, yerr=None,
        aspect_ratio=aspect_ratio, log_xscale=True, log_yscale=True, 
        hist_log_scale=True, write_file_path=None,
        xlabel=None, ylabel=None):
    """
    Makes x-y scatter plot with cumulative histograms for x and y on both axes

    Args:
        x/y (float array): data
        x/yerr (float array, optional): uncertainties
        aspect_ratio (float, optional): figure aspect ratio
        scatter_log_x/yscale (bool, optional): whether to make x/y axes log scale on the scatter plot
        hist_log_scale (bool, optional): whether to make histogram log scale
        write_file_path (str, optional): file path
        x/ylabel (str, optional): labels for write file

    Returns:
        fig, ax1, ax2, ax3 from the figure

    """

    fig = plt.figure(figsize=(8*aspect_ratio, 8))

    ax1 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((3,3), (1,2), colspan=1, rowspan=2, sharey=ax1)

    ### Scatter plot ###
    if((xerr is None) and (yerr is None)):
        ax1.scatter(x, y, marker='o', color=BoiseState_blue)
    if((xerr is None) and (yerr is not None)):
        ax1.errorbar(x, y, yerr=yerr, ls='', marker='o', color=BoiseState_blue)
    if((xerr is not None) and (yerr is not None)):
        ax1.errorbar(x, y, xerr=xerr, yerr=yerr, 
                ls='', marker='o', color=BoiseState_blue)
    if(log_xscale):
        ax1.set_xscale('log')
    if(log_yscale):
        ax1.set_yscale('log')

    # Write plot data out to file
    if(write_file_path is not None):
        # Make sure I sent x/ylabel
        if((xlabel is None) or (ylabel is None)):
                raise ValueError("x/ylabel not given!")

        filename = write_file_path + "_scatterplot.csv"

        if((xerr is None) and (yerr is None)):
            write_out_plot_data(x, y, xlabel, ylabel, filename=filename)
        if((xerr is None) and (yerr is not None)):
            write_out_plot_data(x, y, xlabel, ylabel, yerr=yerr,
                    filename=filename)
        if((xerr is not None) and (yerr is not None)):
            write_out_plot_data(x, y, xlabel, ylabel, xerr=xerr, yerr=yerr,
                    filename=filename)

    ### x cumulative histogram ###
    bins, cum_hst = calc_cum_hist(x)
    ax2.plot(bins, cum_hst, lw=6, color=BoiseState_blue)
    if(hist_log_scale):
        ax2.set_yscale('log')

    # Write plot data out to file
    if(write_file_path is not None):
        # Make sure I sent x/ylabel
        if((xlabel is None) or (ylabel is None)):
                raise ValueError("x/ylabel not given!")

        filename = write_file_path + "_%s_cum_hist.csv" % xlabel

        write_out_plot_data(bins, cum_hst, xlabel, "cum_hist",
                filename=filename)

    ### y cumulative histogram ###
    bins, cum_hst = calc_cum_hist(y)
    ax3.plot(cum_hst, bins, lw=6, color=BoiseState_blue)
    if(hist_log_scale):
        ax3.set_xscale('log')

    # Write plot data out to file
    if(write_file_path is not None):
        # Make sure I sent x/ylabel
        if((xlabel is None) or (ylabel is None)):
                raise ValueError("x/ylabel not given!")

        filename = write_file_path + "_%s_cum_hist.csv" % ylabel
        write_out_plot_data(bins, cum_hst, ylabel, "cum_hist",
                filename=filename)

    return fig, ax1, ax2, ax3

def calc_cum_hist(x, x_lt_value=True):
    """
    Calculate cumulative histogram

    Args:
        x (float array): data
        x_lt_value (bool, optional): whether to plot cumulative histogram for x < value

    Returns:
        bins, cumulative histogram (float arrays)

    """

    srt = np.argsort(x)
    hst, _ = np.histogram(x[srt], bins=[0, *x[srt]])
    bins = x[srt]

    if(x_lt_value):
        return bins, len(x[srt]) - np.cumsum(hst)
    else:
        return bins, np.cumsum(hst)

def extract_sols_from_filenames(filename):
    """
    Returns sol associated with filename

    Args:
        filename (list of strs): file names from which to extract sols
    
    Returns:
        list of sols (list of ints)
    """

    sols = []
    for cur_name in filename:
        ind = cur_name.find("sol_")
        sols.append(int(cur_name[ind+4:ind+8]))

    return sols

def how_many_hours_per_sol(filename):
    """
    Calculate how many hours observations were collected for the sol
    represented in filename while accounting for gaps in files

    Args:
        filename (str): the name of the file

    Returns:
        number of hours (float)
    """

    time, pressure = read_Perseverance_MEDA_data(filename)

    # Break into gaps
    gapped_time, gapped_data = break_at_gaps(time, pressure)

    observed_time = 0.
    for i in range(len(gapped_time)):
        observed_time += (np.max(gapped_time[i]) - np.min(gapped_time[i]))

    return observed_time

def how_much_of_the_hour_observed(filename, hour_edges=None):
    """
    Reads in MEDA data file and calculates how much of each hour in hour_edges
    was observed in that data file

    Args:
        filename (str): path to MEDA data file
        hour_edges (list of floats, optional): edges of hours to analyze

    Returns:
        hour edges and number of the hour observed in the data file 
    """

    # As measured in LTST, there are more seconds per Mars hour than on Earth
    number_of_seconds_in_earth_day = 86400.
    number_of_seconds_in_martian_sol = 88775. 
    how_much_longer_martian_hour = number_of_seconds_in_martian_sol/\
            number_of_seconds_in_earth_day

    if(hour_edges is None):
        hours = np.arange(0, 25)
        hour_edges = np.array([hours[0] - 0.5])

        for hour in hours:
            hour_edges = np.append(hour_edges, hour + 0.5)

    time, pressure = read_Perseverance_MEDA_data(filename)
    # Calculate the time sampling - should be a second each time!
    sampling = mode(time[1:] - time[0:-1]).mode[0]

    how_many_hours_per_hour = np.zeros(len(hour_edges) - 1)
    for i in range(len(hour_edges)-1):
        ind = (time >= hour_edges[i]) & (time <= hour_edges[i+1])

        how_many_hours_per_hour[i] = len(time[ind])*sampling/\
                how_much_longer_martian_hour

    return hour_edges, how_many_hours_per_hour

def get_RDS_data(sol, data_dir="../data/data_calibrated_env/"):
    """
    Returns the time in LTST and RDS dataset for a given sol

    Args:
        sol (int): desired sol
        data_dir (str): path to files

    Returns:
        time, RDS_data
    """

    RDS_data_file = data_dir +\
            "sol_*%i/WE*%i*RDS*.CSV" %\
            (sol, sol)
    
    filename = glob(RDS_data_file)[0]
    
    RDS_data = np.genfromtxt(filename, delimiter=',', 
            names=True, dtype=None)
    time = make_seconds_since_midnight(filename)

    return time, RDS_data

def collect_RDS_points(time, t0, Gamma, num_fwhm=3.):
    """
    Collect RDS data points around a vortex encounter

    Args:
        time (float array): time of RDS data set
        t0 (float): LTST of vortex encounter
        Gamma (float): FWHM (in seconds) of vortex encounter
        num_fwhm (float): number of FWHM to seek values for

    Returns:
        times and data between 1xGamma and 3xGamma before/after and 
        times and data during encounter
    """
    
    # times before or after the encounter
    before_and_after_ind = (((time - t0) < -Gamma/3600.) &\
    ((time - t0) > -num_fwhm*Gamma/3600.)) |\
    (((time - t0) > Gamma/3600.) &\
    ((time - t0) < num_fwhm*Gamma/3600.))

    during_ind = np.abs(time - t0) < Gamma/3600.

    return during_ind, before_and_after_ind

def convert_decimal_to_HMS_time(decimal_time, num_decimal_sec=0):
    """
    Convert decimal time to a string with format HH:MM:SS

    Args:
        decimal_time (float): time as a decimal (e.g., 12:30 is 12.5 in
        decimal)
        num_decimal_sec (optional int): number of decimals to include in the
        seconds field (defaults to zero)

    Returns:
        string of time in HH:MM:SS format 

    """

    HH = int(np.floor(decimal_time))
    MM = int(np.floor((decimal_time - HH)*60))
    # Integer and decimal parts are separate
    SS_int = int(np.floor((decimal_time - HH - MM/60)*60*60))
    SS_dec = (decimal_time - HH - MM/60 - SS_int/60/60)*60*60

    # Dealing with the decimal part of the seconds requires a bit more care.
    # str(SS_dec)[2:] strips off the 0 and decimal from the number.
    if(num_decimal_sec > 0):
        SS_dec_str = "." + str(SS_dec)[2:num_decimal_sec + 2].zfill(num_decimal_sec)
    else:
        # If I don't actually want sub-second precision
        SS_dec_str = ""

    return str(HH).zfill(2) + ":" + str(MM).zfill(2) + ":" +\
            str(SS_int).zfill(2) + SS_dec_str

def how_many_decimals_uncertainty(uncertainty):
    """
    Returns the number of decimal places needed to reflect uncertainty

    Args:
        uncertainty (float): uncertainty
    Returns:
        number of decimals (int)

    """

    ret_val = 0
    if(uncertainty <= 1): 
        ret_val = int(np.abs(np.floor(np.log10(uncertainty))))

    return ret_val
