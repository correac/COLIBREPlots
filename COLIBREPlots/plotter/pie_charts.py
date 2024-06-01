import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import numpy as np
import h5py
from simulation.simulation_data import read_simulation


def read_galactic_abundances(sim_info):

    filename = "./outputs/MWGalaxies_contribution_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        C = file["Data/C"][:]
        N = file["Data/N"][:]
        Mg = file["Data/Mg"][:]
        O = file["Data/O"][:]
        Si = file["Data/Si"][:]
        Fe = file["Data/Fe"][:]
        Ne = file["Data/Ne"][:]

        C_AGB = file["Data/C_AGB"][:]
        N_AGB = file["Data/N_AGB"][:]
        Mg_AGB = file["Data/Mg_AGB"][:]
        O_AGB = file["Data/O_AGB"][:]
        Si_AGB = file["Data/Si_AGB"][:]
        Fe_AGB = file["Data/Fe_AGB"][:]
        Ne_AGB = file["Data/Ne_AGB"][:]

        C_SNII = file["Data/C_SNII"][:]
        N_SNII = file["Data/N_SNII"][:]
        Mg_SNII = file["Data/Mg_SNII"][:]
        O_SNII = file["Data/O_SNII"][:]
        Si_SNII = file["Data/Si_SNII"][:]
        Fe_SNII = file["Data/Fe_SNII"][:]
        Ne_SNII = file["Data/Ne_SNII"][:]

        Fe_SNIa = file["Data/Fe_SNIa"][:]
        HaloID = file["Data/HostHaloID"][:]


    return {"C":C, "N":N, "Mg":Mg, "O":O, "Si":Si, "Ne":Ne, "Fe":Fe,
            "Fe_AGB":Fe_AGB, "C_AGB":C_AGB, "N_AGB":N_AGB, "Mg_AGB":Mg_AGB,
            "O_AGB":O_AGB, "Si_AGB":Si_AGB, "Ne_AGB":Ne_AGB, "Fe_SNIa":Fe_SNIa,
            "Fe_SNII": Fe_SNII, "C_SNII": C_SNII, "N_SNII": N_SNII, "Mg_SNII": Mg_SNII,
            "O_SNII": O_SNII, "Si_SNII": Si_SNII, "Ne_SNII": Ne_SNII, "HaloIndex":HaloID}

def plot_pie(ax, element, data):

    HaloIndx = data["HaloIndex"]
    mass_AGB = data[element+"_AGB"]
    mass_SNII = data[element+"_SNII"]
    total_mass = data[element]
    if element == 'Fe':
        mass_SNIa = data['Fe_SNIa']

    # unique_HaloIndx = np.unique(HaloIndx)
    # num_sample = len(unique_HaloIndx)
    #
    # mass_frac_AGB = np.zeros(num_sample)
    # mass_frac_CCSN = np.zeros(num_sample)
    #
    # for i in range(num_sample):
    #     select = np.where(HaloIndx == unique_HaloIndx[i])[0]
    #     mass_frac_AGB[i] = np.sum(mass_AGB[select]) / np.sum(total_mass[select])
    #     mass_frac_CCSN[i] = np.sum(mass_SNII[select]) / np.sum(total_mass[select])
    #
    # print(np.median(mass_frac_AGB),np.median(mass_frac_CCSN))

    if element == 'Fe':
        masses = [np.sum(mass_AGB) / np.sum(total_mass),
                  np.sum(mass_SNII) / np.sum(total_mass),
                  np.sum(mass_SNIa) / np.sum(total_mass)]
        color_list = ['lightskyblue', 'steelblue', 'khaki']
    else:
        masses = [np.sum(mass_AGB) / np.sum(total_mass),
                  np.sum(mass_SNII) / np.sum(total_mass)]
        color_list = ['lightskyblue', 'steelblue']

    ax.pie(masses, autopct='%1.1f%%', radius=1.2, colors=color_list,
           wedgeprops=dict(edgecolor='w'))

def plot_pie_charts(config_parameters):

    i = 0
    sim_info = read_simulation(config_parameters, i)
    data = read_galactic_abundances(sim_info)

    # Plot parameters
    params = {
        "font.size": 12,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (10, 1.6),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.0,
        "figure.subplot.top": 0.86,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.0,
        "figure.subplot.hspace": 0.0,
    }
    rcParams.update(params)
    plt.figure()

    ax = plt.subplot(1, 7, 1)
    ax.set_title("Carbon")
    plot_pie(ax,"C", data)

    ###
    ax = plt.subplot(1, 7, 2)
    ax.set_title("Nitrogen")
    plot_pie(ax,"N", data)

    ###
    ax = plt.subplot(1, 7, 3)
    ax.set_title("Magnesium")
    plot_pie(ax,"Mg", data)

    ###
    ax = plt.subplot(1, 7, 4)
    ax.set_title("Oxygen")
    plot_pie(ax,"O", data)

    ###
    ax = plt.subplot(1, 7, 5)
    ax.set_title("Silicon")
    plot_pie(ax,"Si", data)

    ###
    ax = plt.subplot(1, 7, 6)
    ax.set_title("Neon")
    plot_pie(ax,"Ne", data)

    ###
    ax = plt.subplot(1, 7, 7)
    ax.set_title("Iron")
    plot_pie(ax,"Fe", data)

    labels = 'AGB', 'CCSN', 'SNIa'
    plt.legend(labels, loc=[-6.6, 0.25], ncol=1, labelspacing=0.1, handlelength=1, handletextpad=0.05,
               frameon=False, fontsize=12, columnspacing=1)

    plt.savefig(config_parameters.output_directory + "pie_charts_MW.png", dpi=300)