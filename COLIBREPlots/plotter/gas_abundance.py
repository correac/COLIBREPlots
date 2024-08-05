"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import numpy as np
import h5py
from simulation.simulation_data import read_simulation
from simulation.utilities import constants
from .observations import plot_Hayden, plot_Nicolls, plot_Berg

def plot_median_relation_one_sigma(x, y, color, output_name):
    num_min_per_bin = 5
    bins = np.arange(7, 10, 0.25)
    ind = np.digitize(x, bins)
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ylo, '-', lw=0.3, color=color, zorder=100)
    plt.plot(xm, yhi, '-', lw=0.3, color=color, zorder=100)
    plt.fill_between(xm, ylo, yhi, color=color, alpha=0.3, edgecolor=None, zorder=0)
    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=100)
    if output_name is not None:
        plt.plot(xm, ym, '-', lw=1.5, color=color, label=output_name, zorder=100)
    else:
        plt.plot(xm, ym, '-', lw=1.5, color=color, zorder=100)


def plot_ArellanoCordova2020():
    # Arellano+ adopt the solar 12+log(O/H) of 8.73±0.04 as recommended by Lodders (2019).
    O_H_Sun_Lodders = 8.73

    file = './plotter/Observational_data/Arellano_Cordova_2020.txt'
    data = np.loadtxt(file, usecols=(1, 2, 3, 4))
    xdata = data[:,0] # Galactocentric radius [ kpc ]
    xerr = data[:, 1]
    ydata = data[:, 2] + O_H_Sun_Lodders - constants.O_H_Sun_Asplund
    yerr = data[:, 3]
    plt.errorbar(xdata, ydata, yerr=yerr, xerr=xerr,
                 mfc='black', ecolor='black', ms=2, mew=0.1, fmt='o', label = 'Arellano-Cordova et al. (2020)')

def read_galactic_abundances(sim_info):

    filename = "./outputs/MWGalaxies_gas_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        C = file["Data/C"][:]
        N = file["Data/N"][:]
        O = file["Data/O"][:]
        Fe = file["Data/Fe"][:]
        H = file["Data/H"][:]
        O_diffuse = file["Data/O_diffuse"][:]
        H_diffuse = file["Data/H_diffuse"][:]
        galactocentric_radius = file["Data/GalactocentricRadius"][:]
        galactocentric_z = file["Data/GalactocentricZ"][:]
        mass = file["Data/Mass"][:]
        temperature = file["Data/Temperature"][:]
        density = file["Data/Density"][:]
        HaloIndex = file["Data/HostHaloID"][:]
        kappa = file["Data/HostGalaxyKappa"][:]

    return {"Fe":Fe, "C":C, "N":N, "O":O, "H":H, "O_diffuse":O_diffuse, "H_diffuse":H_diffuse,
            "GalR":galactocentric_radius, "GalZ":galactocentric_z, "kappa":kappa,
            "mass":mass, "temperature":temperature, "density":density, "HaloIndex":HaloIndex}


def plot_gas_gradient(data, color, output_name):

    GalR = data["GalR"]
    Galz = data["GalZ"]
    HaloIndex = data["HaloIndex"]
    kappa = data["kappa"]
    Oxygen = data["O_diffuse"]
    Hydrogen = data["H_diffuse"]
    mass = data["mass"]
    temperature = data["temperature"]
    nh = data["density"]
    unique_HaloIndx = np.unique(HaloIndex)
    num_sample = len(unique_HaloIndx)

    radial_bins = np.arange(2, 24, 2)
    num_bins = len(radial_bins)-1

    xm = 0.5 * (radial_bins[1:] + radial_bins[:-1])
    ym = np.zeros(num_bins)
    ylo = np.zeros(num_bins)
    yhi = np.zeros(num_bins)

    for i in range(num_bins):
        select_radius = np.where((GalR >= radial_bins[i]) & (GalR < radial_bins[i+1]) & (np.abs(Galz) < 1))[0]

        gradient = []
        for j in range(num_sample):
            select_halo = np.where(HaloIndex[select_radius] == unique_HaloIndx[j])[0]
            select_kappa = np.where(kappa[select_radius[select_halo]] > 0.3)[0]
            # if len(select_halo) == 0: continue
            if len(select_kappa) == 0: continue

            select_particles = select_radius[select_halo[select_kappa]]
            # gas_is_cold_dense = np.where((temperature[select_particles] < 10**4.5) & (nh[select_particles] > 0.1))[0]
            # if len(gas_is_cold_dense) == 0:continue

            gas_mass = mass[select_particles]
            gas_O_over_H_diffuse = Oxygen[select_particles] * constants.mH_in_cgs
            gas_O_over_H_diffuse /= ( constants.mO_in_cgs * Hydrogen[select_particles] )
            gas_O_over_H_diffuse = gas_O_over_H_diffuse * gas_mass
            gas_O_over_H_diffuse = np.sum(gas_O_over_H_diffuse)
            gas_mass = np.sum(gas_mass)
            # gas_O_over_H_diffuse = gas_O_over_H_diffuse[gas_is_cold_dense] * gas_mass[gas_is_cold_dense]
            # gas_O_over_H_diffuse = np.sum(gas_O_over_H_diffuse)
            # gas_mass = np.sum(gas_mass[gas_is_cold_dense])
            log_O_over_H = np.log10(gas_O_over_H_diffuse.value / gas_mass)
            gradient = np.append(gradient, log_O_over_H + 12.)

        if len(gradient) == 0:continue
        ym[i] = np.median(gradient)
        ylo[i] = np.percentile(gradient, 16, axis=0)
        yhi[i] = np.percentile(gradient, 84, axis=0)

    plt.fill_between(xm, ylo, yhi, color=color, alpha=0.2, edgecolor=None, zorder=0)
    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=100)
    plt.plot(xm, ym, '-', lw=1.5, color=color, label=output_name, zorder=100)


def plot_gas_abundance_gradient(config_parameters):

    color_list = ['darkblue','tab:blue','tab:orange','crimson','tab:green']

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (4, 2.5),
        "figure.subplot.left": 0.14,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.05,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_ArellanoCordova2020()
    plt.plot([8.2],[constants.O_H_Sun_Asplund],marker='$\odot$',ms=6, color='black',markeredgewidth=0.01)
    # i = 0
    # sim_info = read_simulation(config_parameters, i)
    # data = read_galactic_abundances(sim_info)
    # plot_gas_gradient(data, color_list[i], sim_info.simulation_name)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_galactic_abundances(sim_info)
        plot_gas_gradient(data, color_list[i], sim_info.simulation_name)

    plt.axis([4, 18, 7, 9.5])
    plt.xlabel("Galactocentric distance [kpc]")
    plt.ylabel("Gas Diffuse 12+$\log_{10}$(O/H)")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0,0.01], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=1, columnspacing=0.1)
    plt.savefig(config_parameters.output_directory + "gas_abundance_gradient_comparison.png", dpi=300)

def plot_gas_CNO(data, color, simulation_name, option):

    # GalR = data["GalR"]
    # Galz = data["GalZ"]
    # HaloIndex = data["HaloIndex"]
    # kappa = data["kappa"]
    Oxygen = data["O"]
    Carbon = data["C"]
    Hydrogen = data["H"]
    Nitrogen = data["N"]
    mass = data["mass"]
    temperature = data["temperature"]
    nh = data["density"]

    gas_is_cold_dense = np.where((temperature < 10**4.5) & (nh > 0.1))[0]

    gas_mass = mass[gas_is_cold_dense]
    gas_O_over_H = Oxygen[gas_is_cold_dense] * constants.mH_in_cgs
    gas_O_over_H /= (constants.mO_in_cgs * Hydrogen[gas_is_cold_dense])
    #gas_O_over_H = gas_O_over_H * gas_mass
    #gas_O_over_H = np.sum(gas_O_over_H)
    #gas_mass = np.sum(gas_mass)
    # log_O_over_H = np.log10(gas_O_over_H.value / gas_mass) + 12.
    log_O_over_H = np.log10(gas_O_over_H.value) + 12.

    if option == "NO":
        gas_N_over_O = Nitrogen[gas_is_cold_dense] * constants.mO_in_cgs
        gas_N_over_O /= (constants.mN_in_cgs * Oxygen[gas_is_cold_dense])
        # gas_N_over_O = gas_N_over_O * gas_mass
        # gas_N_over_O = np.sum(gas_N_over_O)
        # gas_mass = np.sum(gas_mass)
        # log_N_over_O = np.log10(gas_N_over_O.value / gas_mass)
        log_N_over_O = np.log10(gas_N_over_O.value)

        plt.plot(log_O_over_H, log_N_over_O, 'o')
        plot_median_relation_one_sigma(log_O_over_H, log_N_over_O, color, simulation_name)

    if option == "CO":

        gas_C_over_O = Carbon[gas_is_cold_dense] * constants.mO_in_cgs
        gas_C_over_O /= (constants.mC_in_cgs * Oxygen[gas_is_cold_dense])
        # gas_C_over_O = gas_C_over_O * gas_mass
        # gas_C_over_O = np.sum(gas_C_over_O)
        # gas_mass = np.sum(gas_mass)
        # log_C_over_O = np.log10(gas_C_over_O.value / gas_mass)
        log_C_over_O = np.log10(gas_C_over_O.value)

        plt.plot(log_O_over_H, log_C_over_O, 'o')
        plot_median_relation_one_sigma(log_O_over_H, log_C_over_O, color, simulation_name)

def plot_cno_relations(config_parameters):

    color_list = ['steelblue']
    i = 0
    sim_info = read_simulation(config_parameters, i)
    data = read_galactic_abundances(sim_info)

    # Plot parameters
    params = {
        "font.size": 10,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (6, 2.5),
        "figure.subplot.left": 0.14,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.05,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Hayden()
    plot_Nicolls('NO')
    plot_Berg()
    plot_gas_CNO(data, color_list[i], sim_info.simulation_name, "NO")

    plt.axis([7, 10, -2, 0.5])
    plt.ylabel("Gas (Dust+Diffuse) $\log_{10}$(N/O)")
    plt.xlabel("Gas (Dust+Diffuse) 12+$\log_{10}$(O/H)")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0,0.01], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=1, columnspacing=0.1)

    ####

    ax = plt.subplot(1, 2, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Nicolls('CO')
    plot_gas_CNO(data, color_list[i], sim_info.simulation_name, "CO")

    plt.axis([7, 10, -2, 1.0])
    plt.ylabel("Gas (Dust+Diffuse) $\log_{10}$(C/O)")
    plt.xlabel("Gas (Dust+Diffuse) 12+$\log_{10}$(O/H)")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0,0.01], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=1, columnspacing=0.1)


    plt.savefig(config_parameters.output_directory + "gas_abundance_cno.png", dpi=300)
