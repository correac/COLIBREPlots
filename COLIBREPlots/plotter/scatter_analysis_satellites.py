"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import numpy as np
import h5py
import scipy.stats as stat
from simulation.simulation_data import read_simulation
from .observations import (plot_satellites, plot_observations_Kirby_2010, plot_observations_Mead_2024)
from .util import plot_median_relation_one_sigma

def read_MW_abundances(sim_info, element_list):

    filename = "./outputs/MWGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        x = file["Data/"+element_list[0]][:]
        y = file["Data/"+element_list[1]][:]

    return x, y


def read_satellites_abundances(sim_info, element_list):
    filename = "./outputs/SatelliteGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        x = file["Data/" + element_list[0]][:]
        y = file["Data/" + element_list[1]][:]

    return x, y


def read_MW_scatter_in_abundances(sim_info, element_list):

    filename = "./outputs/MWGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        HaloIndx = file["Data/HostHaloID"][:]
        x = file["Data/"+element_list[0]][:]
        y = file["Data/"+element_list[1]][:]

    unique_HaloIndx = np.unique(HaloIndx)
    num_sample = len(unique_HaloIndx)

    sig_x = np.zeros(num_sample)
    sig_y = np.zeros(num_sample)
    for i in range(num_sample):

        select = np.where(HaloIndx == unique_HaloIndx[i])[0]
        Fe_H = x[select]
        low_Z = np.where((Fe_H >= -2) & (Fe_H <= -1))[0]
        sig_y[i] = np.std(y[select[low_Z]])
        sig_x[i] = np.std(x[select[low_Z]])

    return x, y, sig_x, sig_y

def read_satellites_scatter_in_abundances(sim_info, min_metal_cut, max_metal_cut):

    filename = "./outputs/SatelliteGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        HaloIndx = file["Data/HostSubHaloID"][:]
        Fe_H = file["Data/Fe_H"][:]
        O_Fe = file["Data/O_Fe"][:]
        Mg_Fe = file["Data/Mg_Fe"][:]

    unique_HaloIndx = np.unique(HaloIndx)
    num_sample = len(unique_HaloIndx)

    sig_Fe_H = np.zeros(num_sample)
    sig_O_Fe = np.zeros(num_sample)
    sig_Mg_Fe = np.zeros(num_sample)

    for i in range(num_sample):

        select = np.where(HaloIndx == unique_HaloIndx[i])[0]
        low_Z = np.where((Fe_H[select] >= min_metal_cut) & (Fe_H[select] <= max_metal_cut))[0]
        if len(low_Z) < 5: continue
        sig_Fe_H[i] = np.std(Fe_H[select[low_Z]])
        sig_O_Fe[i] = np.std(O_Fe[select[low_Z]])
        sig_Mg_Fe[i] = np.std(Mg_Fe[select[low_Z]])

    return sig_O_Fe, sig_Mg_Fe, sig_Fe_H


def plot_error_bar(i, x, color, id_name):

    err = np.zeros((2, 1))
    err[0, 0] = np.median(x) - np.percentile(x, 16)
    err[1, 0] = np.percentile(x, 84) - np.median(x)
    plt.errorbar(np.array([i]), np.array(np.median(x)), yerr=err, marker='o',
                 markersize=6, color=color, markeredgecolor="white", markeredgewidth=0.5,
                 ls='none', lw=1.5, label=id_name, zorder=100)


def plot_low_Z_distribution(Fe_H, Mg_Fe, color, lt, label, with_label=False):
    select = np.where((Fe_H>=-2) & (Fe_H<= -1))[0]

    bins = np.arange(-2, 2, 0.1)
    hist, bin_edges = np.histogram(Mg_Fe[select], density=True, bins=bins)
    hist /= np.max(hist)
    bin_edges = 0.5 * ( bin_edges[1:] + bin_edges[:-1] )
    if with_label == True:
        plt.plot(bin_edges, hist, color=color, ls=lt, lw=1, label=label)
    else:
        plt.plot(bin_edges, hist, color=color, ls=lt, lw=1)


def plot_scatter_of_MW_satellites(config_parameters):

    color_list = ['steelblue', 'lightskyblue', 'y', 'salmon']

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 4),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.08,
        "figure.subplot.top": 0.98,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(2, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    # plot_MW_data('Mg')
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_MW_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        plot_median_relation_one_sigma(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)

    # plot_GALAH("Mg_Fe")
    plt.axis([-4, 1, -2, 2])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)

    plt.ylabel("[Mg/Fe]")
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.0, 0.75], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=2, columnspacing=0.5)
    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.15, 'MW-type galaxies', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, zorder=1000)

    #########################
    ax = plt.subplot(2, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_satellites('Mg')
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_satellites_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        plot_median_relation_one_sigma(Fe_H, Mg_Fe, color_list[i], None)

    plt.axis([-4, 1, -2, 2])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)

    plt.ylabel("[Mg/Fe]")
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.4, 0.15, 'Satellite galaxies', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, zorder=1000)
    plt.legend(loc=[0.0, 0.82], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=2, columnspacing=0.5)

    #########################
    ax = plt.subplot(2, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_satellites_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        if i == 0:
            plot_low_Z_distribution(Fe_H, Mg_Fe, 'black', '--', 'Satellites galaxies', with_label=True)
            plot_low_Z_distribution(Fe_H, Mg_Fe, color_list[i], '--', 'Satellites galaxies', with_label=False)
        else:
            plot_low_Z_distribution(Fe_H, Mg_Fe, color_list[i], '--', 'Satellites galaxies', with_label=False)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_MW_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        if i == 0:
            plot_low_Z_distribution(Fe_H, Mg_Fe, 'black', '-', 'MW-type galaxies', with_label=True)
            plot_low_Z_distribution(Fe_H, Mg_Fe, color_list[i], '-', 'MW-type galaxies', with_label=False)
        else:
            plot_low_Z_distribution(Fe_H, Mg_Fe, color_list[i], '-', 'MW-type galaxies', with_label=False)

    plt.axis([-2, 1, 0, 1.05])
    plt.ylabel("Normalized Histogram")
    plt.xlabel("[Mg/Fe]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.0, 0.78], labelspacing=0.05, handlelength=1, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=1, columnspacing=0.5)

    #########################
    #########################


    ax = plt.subplot(2, 3, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    delta = -0.01
    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe, sig_Fe, sig_Mg = read_MW_scatter_in_abundances(sim_info,("Fe_H","Mg_Fe"))
        _, O_Fe, _, sig_O = read_MW_scatter_in_abundances(sim_info,("Fe_H","O_Fe"))

        plot_error_bar(1 + delta, sig_Mg, color_list[i], None)
        plot_error_bar(2 + delta, sig_O, color_list[i], None)
        delta += 0.01

    # plot_APOGEE_scatter(("Fe_H","Mg_Fe"),True)
    # plot_APOGEE_scatter(("Fe_H","O_Fe"), False)
    # plot_MW_scatter("Mg",True)
    # plot_MW_scatter("O",False)

    plt.axis([0, 3, 0, 1])
    xticks = np.array([1, 2])
    labels = ["[O/Fe]", "[Mg/Fe]"]
    plt.xticks(xticks, labels)
    plt.ylabel("Scatter")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.04, 0.65], labelspacing=0.05, handlelength=0.3, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=1, columnspacing=0.5)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.1, 0.95, '$-2\le$ [Fe/H]$\le -1$ $|$ Milky Way', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=props, zorder=1000)

    #########################
    ax = plt.subplot(2, 3, 5)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_observations_Mead_2024()

    delta = -0.01
    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        sO_Fe, sMg_Fe, sFe_H = read_satellites_scatter_in_abundances(sim_info, -2, -1)
        plot_error_bar(1+delta, sO_Fe, color_list[i], None)
        plot_error_bar(2+delta, sMg_Fe, color_list[i], None)
        delta += 0.01

    plt.axis([0, 3, 0, 1])
    xticks = np.array([1, 2])
    labels = ["[O/Fe]", "[Mg/Fe]"]
    plt.xticks(xticks, labels)
    plt.ylabel("Scatter")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.04,0.66],labelspacing=0.05, handlelength=0.3, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=3, columnspacing=0.2)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.05, 0.95, '$-2\le$ [Fe/H]$\le -1$ $|$ Mead+ (2024)', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=props, zorder=1000)

    #########################
    ax = plt.subplot(2, 3, 6)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_observations_Kirby_2010()

    delta = -0.01
    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        _, _, sFe_H = read_satellites_scatter_in_abundances(sim_info,-2.5,0.5)

        plot_error_bar(1+delta, sFe_H, color_list[i], None)
        delta += 0.01

    plt.axis([0, 2, 0, 1])
    xticks = np.array([1])
    labels = ["[Fe/H]"]
    plt.xticks(xticks, labels)
    plt.ylabel("Scatter")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.7,0.25],labelspacing=0.05, handlelength=0.3, handletextpad=0.3,
               frameon=False, fontsize=8, ncol=1, columnspacing=0.2)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.03, 0.95, '$-2.5\le$ [Fe/H]$\le 0.5$ $|$ Kirby+ (2010)', transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=props, zorder=1000)

    plt.savefig(config_parameters.output_directory + "Scatter.png", dpi=300)

#
# def plot_histogram(x, radius, z, color, label):
#
#     bins = np.arange(-2,1,0.1)
#     select_zcoord = np.where(np.abs(z)<1)[0]
#     inner = np.where(radius[select_zcoord] <= 5)[0]
#     weights = 1./np.ones(len(inner))
#     inhist, bin_edges, _ = stat.binned_statistic(x=x[select_zcoord[inner]], values=weights, statistic="sum", bins=bins, )
#     inhist /= np.max(inhist)
#
#     middle = np.where((radius[select_zcoord] > 5) & (radius[select_zcoord] <= 9))[0]
#     weights = 1./np.ones(len(middle))
#     midhist, _, _ = stat.binned_statistic(x=x[select_zcoord[middle]], values=weights, statistic="sum", bins=bins, )
#     midhist /= np.max(midhist)
#
#     outer = np.where(radius[select_zcoord] > 9)[0]
#     weights = 1./np.ones(len(outer))
#     outhist, _, _ = stat.binned_statistic(x=x[select_zcoord[outer]], values=weights, statistic="sum", bins=bins, )
#     outhist /= np.max(outhist)
#
#     bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
#     plt.plot(bin_centers, inhist, '-', lw=1, color=color)
#     plt.plot(bin_centers, midhist, '-', lw=2, color=color)
#     plt.plot(bin_centers, outhist, '-', lw=3, color=color)
#


# def plot_scatter_MW_with_radius(config_parameters):
#
#     color_list = ['tab:blue','tab:orange','crimson','darkblue']
#
#     # Plot parameters
#     params = {
#         "font.size": 11,
#         "font.family": "Times",
#         "text.usetex": True,
#         "figure.figsize": (7, 2.2),
#         "figure.subplot.left": 0.07,
#         "figure.subplot.right": 0.98,
#         "figure.subplot.bottom": 0.18,
#         "figure.subplot.top": 0.95,
#         "lines.markersize": 0.5,
#         "lines.linewidth": 0.2,
#         "figure.subplot.wspace": 0.3,
#         "figure.subplot.hspace": 0.3,
#     }
#     rcParams.update(params)
#     plt.figure()
#     ax = plt.subplot(1, 3, 1)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     for i in range(config_parameters.number_of_inputs):
#
#         sim_info = read_simulation(config_parameters, i)
#         Fe_H, _, radius, z = calculate_MW_abundances(sim_info)
#         plot_histogram(Fe_H, radius, z, color_list[i], sim_info.simulation_name)
#
#     plt.axis([-2, 1, 0, 1.2])
#     # xticks = np.array([-4, -3, -2, -1, 0, 1])
#     # labels = ["-4","-3","-2","-1","0","1"]
#     # plt.xticks(xticks, labels)
#
#     plt.ylabel("PDF")
#     plt.xlabel("[Fe/H]")
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.legend(loc=[0.03,0.8],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
#                frameon=False, fontsize=9, ncol=2, columnspacing=0.23)
#
#     ####
#     ax = plt.subplot(1, 3, 2)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     for i in range(config_parameters.number_of_inputs):
#         sim_info = read_simulation(config_parameters, i)
#         _, Mg_Fe, radius, z = calculate_MW_abundances(sim_info)
#         plot_histogram(Mg_Fe, radius, z, color_list[i], sim_info.simulation_name)
#
#     plt.axis([-1, 1, 0, 1.2])
#     # xticks = np.array([-4, -3, -2, -1, 0, 1])
#     # labels = ["-4", "-3", "-2", "-1", "0", "1"]
#     # plt.xticks(xticks, labels)
#
#     plt.ylabel("PDF")
#     plt.xlabel("[Mg/Fe]")
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
#                frameon=False, fontsize=9, ncol=2, columnspacing=0.23)
#
#     ####
#     ax = plt.subplot(1, 3, 3)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     for i in range(config_parameters.number_of_inputs):
#
#         sim_info = read_simulation(config_parameters, i)
#         Fe_H, Mg_Fe, _, _ = calculate_MW_abundances(sim_info)
#         plot_median_relation(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)
#
#     plt.axis([-4, 1, -2, 2])
#     xticks = np.array([-4, -3, -2, -1, 0, 1])
#     labels = ["-4","-3","-2","-1","0","1"]
#     plt.xticks(xticks, labels)
#
#     plt.ylabel("[Mg/Fe]")
#     plt.xlabel("[Fe/H]")
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.legend(loc=[0.03,0.8],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
#                frameon=False, fontsize=9, ncol=2, columnspacing=0.23)
#
#
#     plt.savefig(config_parameters.output_directory + "MW_radial_Scatter.png", dpi=300)
