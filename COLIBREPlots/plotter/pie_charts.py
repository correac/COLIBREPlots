import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import numpy as np
import h5py
from simulation.simulation_data import read_simulation
from simulation.utilities import constants

def read_galactic_abundances(sim_info):

    filename = "./outputs/MWGalaxies_contribution_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        H = file["Data/H"][:]
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
        Ne_SNII = file["Data/Ne_SNII"][:]
        Fe_SNII = file["Data/Fe_SNII"][:]

        C_SNIa = file["Data/C_SNIa"][:]
        N_SNIa = file["Data/N_SNIa"][:]
        Mg_SNIa = file["Data/Mg_SNIa"][:]
        O_SNIa = file["Data/O_SNIa"][:]
        Si_SNIa = file["Data/Si_SNIa"][:]
        Ne_SNIa = file["Data/Ne_SNIa"][:]
        Fe_SNIa = file["Data/Fe_SNIa"][:]
        HaloID = file["Data/HostHaloID"][:]

        GalR = file["Data/GalactocentricRadius"][:]
        GalZ = file["Data/GalactocentricZ"][:]
        Kappa = file["Data/HostGalaxyKappa"][:]
        Mstellar = file["Data/HostGalaxylogMstellar"][:]


    return {"H":H, "C":C, "N":N, "Mg":Mg, "O":O, "Si":Si, "Ne":Ne, "Fe":Fe,
            "Fe_AGB":Fe_AGB, "C_AGB":C_AGB, "N_AGB":N_AGB, "Mg_AGB":Mg_AGB,
            "O_AGB":O_AGB, "Si_AGB":Si_AGB, "Ne_AGB":Ne_AGB,
            "Fe_SNII": Fe_SNII, "C_SNII": C_SNII, "N_SNII": N_SNII, "Mg_SNII": Mg_SNII,
            "O_SNII": O_SNII, "Si_SNII": Si_SNII, "Ne_SNII": Ne_SNII,
            "Fe_SNIa": Fe_SNIa, "C_SNIa": C_SNIa, "N_SNIa": N_SNIa, "Mg_SNIa": Mg_SNIa,
            "O_SNIa": O_SNIa, "Si_SNIa": Si_SNIa, "Ne_SNIa": Ne_SNIa,
            "HaloIndex":HaloID, "GalR": GalR, "GalZ":GalZ, "Kappa":Kappa, "Mstellar":Mstellar}

def plot_pie(ax, element, data):

    print(element)

    low_limit = 1e-8
    HaloIndx = data["HaloIndex"]
    Kappa = data["Kappa"]
    Mstellar = data["Mstellar"]

    # Let's do a pre-selection based on stellar mass and morphology, shall we?
    # min_mass = np.log10(1e10)
    # max_mass = np.log10(6e10)
    # select_mass = np.where((Mstellar>min_mass) & (Mstellar<max_mass))[0]
    # select_morpho = np.where(Kappa[select_mass]>0.3)[0]
    # HaloIndx = HaloIndx[select_mass[select_morpho]]

    mass_AGB = data[element+"_AGB"]
    mass_SNII = data[element+"_SNII"]
    mass_SNIa = data[element+"_SNIa"]
    total_mass = data[element]
    mass_Fe = data["Fe"]
    mass_H = data["H"]
    GalR = data["GalR"]
    GalZ = data["GalZ"]

    unique_HaloIndx = np.unique(HaloIndx)
    num_sample = len(unique_HaloIndx)
    print("Num sample:", num_sample)

    mass_frac_AGB = np.zeros(num_sample)
    mass_frac_CCSN = np.zeros(num_sample)
    mass_frac_SNIa = np.zeros(num_sample)
    FeH_mean = np.zeros(num_sample)

    for i in range(num_sample):

        select = np.where(HaloIndx == unique_HaloIndx[i])[0]

        R = GalR[select]
        z = GalZ[select]

        select_within_disc = np.where((R>9) & (np.abs(z)>2))[0]
        select = select[select_within_disc]

        FeH_stars = np.clip(mass_Fe[select], low_limit, None) / mass_H[select]
        FeH_stars = np.log10(FeH_stars) - constants.Fe_H_Sun
        FeH_mean[i] = np.mean(FeH_stars)

        mass_frac_AGB[i] = np.sum(mass_AGB[select]) / np.sum(total_mass[select])
        mass_frac_CCSN[i] = np.sum(mass_SNII[select]) / np.sum(total_mass[select])
        mass_frac_SNIa[i] = np.sum(mass_SNIa[select]) / np.sum(total_mass[select])

        if element == "C": # Some of it lost in dust destruction, so no account for it here (?)
            mass_frac_AGB[i] = 1. - mass_frac_SNIa[i] - mass_frac_CCSN[i]

    print(np.mean(FeH_mean),np.median(FeH_mean))
    print(np.mean(mass_frac_AGB), np.mean(mass_frac_CCSN), np.mean(mass_frac_SNIa))
    print(np.mean(mass_frac_AGB) + np.mean(mass_frac_CCSN) + np.mean(mass_frac_SNIa))
    print('=======')

    masses = [
        round(np.mean(mass_frac_AGB),3), round(np.mean(mass_frac_CCSN),3), round(np.mean(mass_frac_SNIa),3)
    ]

    # Add round-off correction:
    if np.sum(masses)<1:
        masses[2] = 1-masses[0]-masses[1]

    color_list = ['lightskyblue', 'steelblue', 'khaki']

    if element == "C":
        FeH = np.mean(FeH_mean)
        text = r'$\langle[\mathrm{Fe}/\mathrm{H}]\rangle{=}$'+'{:.2f}'.format(FeH)
        props = dict(facecolor='white', edgecolor='none', pad=0.0)
        ax.text(-0.58, 1.18, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        text = r'$R_{\mathrm{Gal}}{>}9$ kpc'
        ax.text(-0.58, 1.04, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        text = r'$|z_{\mathrm{Gal}}|{>}2$ kpc'
        ax.text(-0.6, 0.93, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

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
    plt.legend(labels, loc=[-6.6, 0.15], ncol=1, labelspacing=0.1, handlelength=1, handletextpad=0.05,
               frameon=False, fontsize=12, columnspacing=1)

    plt.savefig(config_parameters.output_directory + "pie_charts_MW.png", dpi=300)