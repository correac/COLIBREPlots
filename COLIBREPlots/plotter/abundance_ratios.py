"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import numpy as np
import h5py
from simulation.simulation_data import read_simulation
from .util import plot_median_relation, plot_median_relation_one_sigma
from .observations import (plot_APOGEE, plot_GALAH, plot_Sit2024, plot_StrontiumObsData,
                          plot_data_Zepeda2022, plot_data_Gudin2021, plot_data_Norfolk2019)


def read_galactic_abundances(sim_info):

    filename = "./outputs/MWGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        C_Fe = file["Data/C_Fe"][:]
        N_Fe = file["Data/N_Fe"][:]
        Mg_Fe = file["Data/Mg_Fe"][:]
        O_Fe = file["Data/O_Fe"][:]
        Si_Fe = file["Data/Si_Fe"][:]
        Fe_H = file["Data/Fe_H"][:]
        Ne_Fe = file["Data/Ne_Fe"][:]
        # Ba_Fe = file["Data/Ba_Fe"][:]
        # Sr_Fe = file["Data/Sr_Fe"][:]
        # Eu_Fe = file["Data/Eu_Fe"][:]

    # Mg_H = Mg_Fe + Fe_H
    # Ba_Mg = Ba_Fe - Mg_Fe
    # Sr_Mg = Sr_Fe - Mg_Fe
    # Eu_Mg = Eu_Fe - Mg_Fe

    return {"Fe_H":Fe_H, "C_Fe":C_Fe, "N_Fe":N_Fe, "Mg_Fe":Mg_Fe, "O_Fe":O_Fe,
            "Si_Fe":Si_Fe, "Ne_Fe":Ne_Fe}#, "Ba_Fe":Ba_Fe, "Sr_Fe":Sr_Fe, "Eu_Fe":Eu_Fe,
            #"Mg_H":Mg_H, "Ba_Mg":Ba_Mg, "Sr_Mg":Sr_Mg, "Eu_Mg":Eu_Mg}


def plot_abundance_ratios_comparison(config_parameters):

    color_list = ['steelblue', 'lightskyblue', 'y', 'salmon']

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 4),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.12,
        "figure.subplot.top": 0.98,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.05,
        "figure.subplot.hspace": 0.05,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(2, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","C_Fe"))
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["C_Fe"], color_list[i], sim_info.simulation_name)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[X/Fe]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    plt.legend(loc=[0,0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    plt.plot([-1.91,-1.78], [0.93,0.93], '-', lw=0.5, color='black')

    props = dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.2)
    ax.text(0.72, 0.15, '[C/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","N_Fe"))
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["N_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.text(0.72, 0.15, '[N/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","Mg_Fe"))
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["Mg_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.text(0.67, 0.15, '[Mg/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","O_Fe"))
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["O_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[X/Fe]")
    plt.xticks([-2, -1, 0, 0.5])
    plt.yticks(np.arange(-1,1,0.5))
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.72, 0.15, '[O/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 5)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","Si_Fe"))
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["Si_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.yticks(np.arange(-1,1,0.5))
    plt.xticks([-2, -1, 0, 0.5])
    ax.get_yaxis().set_ticklabels([])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.71, 0.15, '[Si/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 6)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_median_relation_one_sigma(data["Fe_H"], data["Ne_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.yticks(np.arange(-1,1,0.5))
    plt.xticks([-2, -1, 0, 0.5, 1])
    ax.get_yaxis().set_ticklabels([])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.68, 0.15, '[Ne/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.savefig(config_parameters.output_directory + "stellar_abundance_ratio_comparison.png", dpi=300)



def plot_abundance_ratios_single_run(config_parameters):

    color_list = ['steelblue']
    i = 0
    sim_info = read_simulation(config_parameters, i)
    data = read_galactic_abundances(sim_info)

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 4),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.12,
        "figure.subplot.top": 0.98,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.05,
        "figure.subplot.hspace": 0.05,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(2, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","C_Fe"))
    plot_median_relation(data["Fe_H"], data["C_Fe"], color_list[i], sim_info.simulation_name)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[X/Fe]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    plt.legend(loc=[0,0.75], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=1, columnspacing=0.1)

    plt.plot([-1.91,-1.78], [0.83,0.83], '-', lw=0.5, color='black')

    props = dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.2)
    ax.text(0.72, 0.15, '[C/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","N_Fe"))
    plot_median_relation(data["Fe_H"], data["N_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.text(0.72, 0.15, '[N/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","Mg_Fe"))
    plot_median_relation(data["Fe_H"], data["Mg_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.text(0.67, 0.15, '[Mg/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H","O_Fe"))
    plot_median_relation(data["Fe_H"], data["O_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[X/Fe]")
    plt.xticks([-2, -1, 0, 0.5])
    plt.yticks(np.arange(-1,1,0.5))
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.72, 0.15, '[O/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 5)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_APOGEE(("Fe_H", "Si_Fe"))
    plot_median_relation(data["Fe_H"], data["Si_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.yticks(np.arange(-1,1,0.5))
    plt.xticks([-2, -1, 0, 0.5])
    ax.get_yaxis().set_ticklabels([])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.71, 0.15, '[Si/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(2, 3, 6)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_median_relation(data["Fe_H"], data["Ne_Fe"], color_list[i], None)

    plt.axis([-2, 1, -1, 1])
    plt.xlabel("[Fe/H]")
    plt.yticks(np.arange(-1,1,0.5))
    plt.xticks([-2, -1, 0, 0.5, 1])
    ax.get_yaxis().set_ticklabels([])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.68, 0.15, '[Ne/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.savefig(config_parameters.output_directory + "stellar_abundance_ratio.png", dpi=300)

    #######
    #######
    #######

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 2),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.22,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.05,
        "figure.subplot.hspace": 0.05,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_GALAH("Ba_Fe", False)
    plot_data_Gudin2021("Fe_H", "Ba_Fe", True)
    plot_data_Zepeda2022("Fe_H", "Ba_Fe", True)
    # plot_data_Norfolk2019("Fe_H", "Ba_Fe")
    plot_median_relation(data["Fe_H"], data["Ba_Fe"], color_list[i], sim_info.simulation_name)

    plt.axis([-4, 1, -2, 3])
    plt.xticks([-4, -3, -2, -1, 0])
    plt.xlabel("[Fe/H]")
    plt.ylabel("[X/Fe]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.0, 0.81], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=8, ncol=2, columnspacing=0.8)
    ax.text(0.68, 0.15, '[Ba/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.annotate('GALAH data', (-3.72, 1.98), fontsize=8)
    plt.plot([-3.9, -3.72], [2.1, 2.1], '-', lw=0.5, color='black')

    ###
    ax = plt.subplot(1, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_StrontiumObsData()
    plot_data_Gudin2021("Fe_H", "Sr_Fe", False)
    plot_data_Zepeda2022("Fe_H", "Sr_Fe", False)
    # plot_data_Norfolk2019("Fe_H", "Sr_Fe")
    plot_median_relation(data["Fe_H"], data["Sr_Fe"], color_list[i], None)

    plt.axis([-4, 1, -2, 3])
    plt.xlabel("[Fe/H]")
    plt.xticks([-4, -3, -2, -1, 0])
    ax.get_yaxis().set_ticklabels([])
    plt.legend(loc=[0, 0.81], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=8, ncol=2, columnspacing=0.2)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.68, 0.15, '[Sr/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(1, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_GALAH("Eu_Fe",False)
    plot_data_Gudin2021("Fe_H", "Eu_Fe", False)
    plot_data_Zepeda2022("Fe_H", "Eu_Fe", False)
    plot_median_relation(data["Fe_H"], data["Eu_Fe"], color_list[i], None)

    plt.axis([-4, 1, -2, 3])
    plt.xlabel("[Fe/H]")
    plt.xticks([-4, -3, -2, -1, 0, 1])
    ax.get_yaxis().set_ticklabels([])
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.47, 0.81], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=8, ncol=1, columnspacing=0.1)
    ax.text(0.68, 0.15, '[Eu/Fe]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.savefig(config_parameters.output_directory + "stellar_abundance_ratio_rs_processes.png", dpi=300)

    ####

    plt.figure()
    ax = plt.subplot(1, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_data_Zepeda2022("Mg_H", "Ba_Mg", False)
    plot_median_relation(data["Mg_H"], data["Ba_Mg"], color_list[i], sim_info.simulation_name)

    plt.axis([-4, 1, -2, 2])
    plt.xticks([-4, -3, -2, -1, 0])
    plt.xlabel("[Mg/H]")
    plt.ylabel("[X/Mg]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.legend(loc=[0, 0.85], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=1, columnspacing=0.1)
    ax.text(0.68, 0.15, '[Ba/Mg]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ###
    ax = plt.subplot(1, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_data_Zepeda2022("Mg_H", "Sr_Mg", False)
    plot_median_relation(data["Mg_H"], data["Sr_Mg"], color_list[i], None)

    plt.axis([-4, 1, -2, 2])
    plt.xticks([-4, -3, -2, -1, 0])
    plt.xlabel("[Mg/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.text(0.68, 0.15, '[Sr/Mg]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.get_yaxis().set_ticklabels([])

    ###
    ax = plt.subplot(1, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_data_Zepeda2022("Mg_H", "Eu_Mg", False)
    plot_median_relation(data["Mg_H"], data["Eu_Mg"], color_list[i], None)

    plt.axis([-4, 1, -2, 2])
    plt.xticks([-4, -3, -2, -1, 0, 1])
    plt.xlabel("[Mg/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.legend(loc=[0, 0.85], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=1, columnspacing=0.1)
    ax.text(0.68, 0.15, '[Eu/Mg]', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.get_yaxis().set_ticklabels([])

    plt.savefig(config_parameters.output_directory + "stellar_abundance_ratio_MgH_rs_processes.png", dpi=300)