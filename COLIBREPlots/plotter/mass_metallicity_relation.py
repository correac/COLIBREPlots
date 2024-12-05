"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""
import matplotlib.pylab as plt
import matplotlib
from matplotlib.pylab import rcParams
import numpy as np
import h5py
from scipy import interpolate
from simulation.simulation_data import read_simulation
from simulation.utilities.constants import Zsolar
from .observations import (plot_Zahid2017, plot_gallazi, plot_Kudritzki,
                           plot_Yates, plot_Kirby, plot_Panter_2018, plot_Fraser,
                           plot_Tremonti, plot_Andrews, plot_Curti, plot_SAGA,
                           plot_Gallazzi_2021, plot_Romero_Gomez_2023,
                           plot_Romero_Gomez_2023_dwarfs)


def plot_median_relation_one_sigma(x, y, color, output_name, with_scatter, min_mass, max_mass):

    # Let's remove painted nans
    nonan = np.where(y > 0)[0]
    x = x[nonan]
    y = y[nonan]

    if min_mass == 0: min_mass = 6
    if max_mass == 0: max_mass =  13

    num_min_per_bin = 2
    bins = np.arange(min_mass, max_mass, 0.25)
    bins = 10**bins
    ind = np.digitize(x, bins)
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]

    if with_scatter == 1:
        plt.plot(xm, ylo, '-', lw=0.3, color=color, zorder=10000)
        plt.plot(xm, yhi, '-', lw=0.3, color=color, zorder=10000)
        plt.fill_between(xm, ylo, yhi, color=color, alpha=0.3, edgecolor=None, zorder=10000)

    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=10000)
    plt.plot(xm, ym, '-', lw=1.5, color=color, zorder=10000)

    for i in range(len(xm)):print(np.log10(xm[i]), ym[i], ym[i]-ylo[i], yhi[i]-ym[i])
    f = interpolate.interp1d(xm, ym)
    print('1e10',f(1e10))

def read_data(sim_info):

    filename = "./outputs/Galaxies_mass_metallicity_relation_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        FeH = file["Data/Fe_H"][:]
        Z = file["Data/Metallicity"][:]
        OH_diffuse = file["Data/O_H_gas_diffused"][:]
        OH_total = file["Data/O_H_gas_total"][:]
        Mstellar = file["Data/GalaxylogMstellar"][:]

    return {"FeH": FeH, "Z": Z, "OH_diffused": OH_diffuse, "OH_total": OH_total,
            "Mstellar": 10**Mstellar}


def plot_mass_metallicity_relation(config_parameters):

    color_list = ['steelblue','darkblue', 'y', 'salmon']
    # color_list = ['darkblue', 'steelblue', 'y', 'salmon']
    option = [1, 0, 0, 1] # 1 yes, 0 no
    option_min_mass = [7.8, 7.8, 7.8, 7.8]
    option_max_mass = [0, 0, 0, 0]

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.38,
        "figure.subplot.hspace": 0.38,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.2)

    plot_Zahid2017()
    plot_gallazi()
    plot_Kudritzki()
    plot_Yates()
    # plot_Kirby()
    plot_Panter_2018()

    print('Total metallicity')

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["Mstellar"], data["Z"]/Zsolar,
                                       color=color_list[i],
                                       output_name=sim_info.simulation_name,
                                       with_scatter=option[i],
                                       min_mass=option_min_mass[i],
                                       max_mass=option_max_mass[i])

    plt.axis([1e7, 1e12, 5e-2, 5])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$Z_{*}$ [Z$_{\odot}$]")
    # plt.yticks([1e-2, 1e-1, 1, 5], ['$10^{-2}$', '$10^{-1}$', '1','5'])
    # plt.xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
    #            ['$10^{6}$', '$10^{7}$', '$10^{8}$','$10^{9}$','$10^{10}$','$10^{11}$','$10^{12}$'])
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.01,0.64],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=11, ncol=1, columnspacing=0.23)

    ax2 = ax.twinx()
    ax2.axis('off')
    simulation_list = ["No diffusion","Low diffusion","Default diffusion","High diffusion"]
    for i in range(config_parameters.number_of_inputs):
        ax2.plot([], [], lw=2, color=color_list[i], label=simulation_list[i])

    ax2.legend(loc=[0.2, 0.02], ncol=2, labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, facecolor='goldenrod', framealpha=0.3, fontsize=11, columnspacing=1,
               numpoints=1)

    plt.savefig(config_parameters.output_directory + "Mass_Z_relation.png", dpi=300)

    #####

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Zahid2017()
    plot_gallazi()
    plot_Kudritzki()
    plot_Yates()
    # plot_Kirby()
    plot_Panter_2018()

    print('[FeH] relation')

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["Mstellar"], data["FeH"],
                                       color=color_list[i],
                                       output_name=sim_info.simulation_name,
                                       with_scatter=option[i],
                                       min_mass=option_min_mass[i],
                                       max_mass=option_max_mass[i])

    plt.axis([1e7, 1e12, 5e-2, 5])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$M_{*}$ [M$_{\odot}$]")
    plt.ylabel("Stellar (Fe/H) [in units of (Fe/H)$_{\odot}$]")
    # plt.yticks([1e-2, 1e-1, 1, 5], ['$10^{-2}$', '$10^{-1}$', '1','5'])
    # plt.xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
    #            ['$10^{6}$', '$10^{7}$', '$10^{8}$','$10^{9}$','$10^{10}$','$10^{11}$','$10^{12}$'])
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.legend(loc=[0.01,0.59],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
    #            frameon=False, fontsize=11, ncol=1, columnspacing=0.23)

    plt.savefig(config_parameters.output_directory + "Mass_FeH_relation.png", dpi=300)

    #####

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Fraser()
    plot_Tremonti()
    plot_Andrews()
    plot_Curti()
    plot_SAGA()

    print('OH_diffused')


    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["Mstellar"], data["OH_diffused"],
                                       color=color_list[i],
                                       output_name=sim_info.simulation_name,
                                       with_scatter=option[i],
                                       min_mass=option_min_mass[i],
                                       max_mass=option_max_mass[i])

    plt.axis([1e7, 1e12, 7.5, 9.5])
    plt.xscale('log')
    plt.xlabel("$M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$12+\log_{10}$(O/H) (Diffuse)")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
    #            ['$10^{6}$', '$10^{7}$', '$10^{8}$', '$10^{9}$', '$10^{10}$', '$10^{11}$', '$10^{12}$'])
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.legend(loc=[0.01,0.65],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=11, ncol=1, columnspacing=0.23)

    ax2 = ax.twinx()
    ax2.axis('off')
    # simulation_list = ["L050N752","L025N752","L025N376","L012N376"]
    simulation_list = ["No diffusion","Low diffusion","Default diffusion","High diffusion"]
    for i in range(config_parameters.number_of_inputs):
        ax2.plot([], [], lw=2, color=color_list[i], label=simulation_list[i])

    ax2.legend(loc=[0.2, 0.02], ncol=2, labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, facecolor='goldenrod', framealpha=0.3, fontsize=11, columnspacing=1,
               numpoints=1)

    plt.savefig(config_parameters.output_directory + "Mass_OH_diffuse_relation.png", dpi=300)

    #####

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Fraser()
    plot_Tremonti()
    plot_Andrews()
    plot_Curti()
    plot_SAGA()

    print("OH total")

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["Mstellar"], data["OH_total"],
                                       color=color_list[i],
                                       output_name=sim_info.simulation_name,
                                       with_scatter=option[i],
                                       min_mass=option_min_mass[i],
                                       max_mass=option_max_mass[i])

    plt.axis([1e7, 1e12, 7.5, 9.5])
    plt.xscale('log')
    plt.xlabel("$M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$12+\log_{10}$(O/H) (Dust + Diffuse)")
    # plt.xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
    #            ['$10^{6}$', '$10^{7}$', '$10^{8}$', '$10^{9}$', '$10^{10}$', '$10^{11}$', '$10^{12}$'])

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    # plt.legend(loc=[0.01,0.7],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
    #            frameon=False, fontsize=11, ncol=1, columnspacing=0.23)

    plt.savefig(config_parameters.output_directory + "Mass_OH_total_relation.png", dpi=300)

def read_MgFe_data(sim_info):

    filename = "./outputs/Galaxies_mass_metallicity_relation_" + sim_info.simulation_name + ".hdf5"
    # filename = "./outputs/Galaxies_MgFe_mass_relation_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        MgFe = file["Data/Mg_Fe"][:]
        Mstellar = file["Data/GalaxylogMstellar"][:]
        GalaxyType = file["Data/HaloType"][:]

    centrals = np.where(GalaxyType == 10)[0]
    return {"MgFe": MgFe[centrals], "Mstellar": 10**Mstellar[centrals]}

def plot_MgFe_mass_relation(config_parameters):

    color_list = ['steelblue','darkblue', 'y', 'salmon']
    option = [1, 0, 0, 1] # 1 yes, 0 no
    option_min_mass = [7.8, 0, 7.8  , 0]
    option_max_mass = [0, 0, 0, 8.5]

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.38,
        "figure.subplot.hspace": 0.38,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.2)

    plot_Gallazzi_2021()
    plot_Romero_Gomez_2023()
    plot_Romero_Gomez_2023_dwarfs()

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = read_MgFe_data(sim_info)
        plot_median_relation_one_sigma(data["Mstellar"], data["MgFe"],
                                       color=color_list[i],
                                       output_name=sim_info.simulation_name,
                                       with_scatter=option[i],
                                       min_mass=option_min_mass[i],
                                       max_mass=option_max_mass[i])

    plt.axis([5e7, 5e12, -0.5, 0.7])
    plt.xscale('log')
    plt.xlabel("$M_{*}$ [M$_{\odot}$]")
    plt.ylabel("[Mg/Fe]")
    plt.xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12],
               ['$10^{6}$', '$10^{7}$', '$10^{8}$','$10^{9}$','$10^{10}$','$10^{11}$','$10^{12}$'])
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.01,0.76],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=11, ncol=1, columnspacing=0.23)

    ax2 = ax.twinx()
    ax2.axis('off')
    simulation_list = ["L050N752","L025N752","L025N376","L012N376"]
    for i in range(config_parameters.number_of_inputs):
        ax2.plot([], [], lw=2, color=color_list[i], label=simulation_list[i])

    ax2.legend(loc=[0.7, 0.02], ncol=1, labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, facecolor='goldenrod', framealpha=0.3, fontsize=11, columnspacing=1,
               numpoints=1)

    plt.savefig(config_parameters.output_directory + "Mass_MgFe_relation.png", dpi=300)