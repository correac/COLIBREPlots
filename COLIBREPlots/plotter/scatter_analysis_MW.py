"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from tqdm import tqdm
import scipy.stats as stat
from simulation.simulation_data import read_simulation
from simulation import particle_data
from simulation.halo_catalogue import calculate_morphology
from .util import plot_median_relation, plot_median_relation_one_sigma, plot_scatter
import h5py
from simulation.utilities import constants
from .abundance_ratios import plot_APOGEE
from .observations import (plot_APOGEE_scatter, plot_GALAH, plot_MW_data, plot_MW_scatter,
                           plot_Pristine, plot_Pristine_scatter, plot_MW_scatter, plot_APOGEE_scatter)

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


def plot_error_bar(i, x, color, id_name):

    err = np.zeros((2, 1))
    err[0, 0] = np.median(x) - np.percentile(x, 16)
    err[1, 0] = np.percentile(x, 84) - np.median(x)
    plt.errorbar(np.array([i]), np.array(np.median(x)), yerr=err, marker='o',
                 markersize=6, color=color, markeredgecolor="white", markeredgewidth=0.5,
                 ls='none', lw=1.5, label=id_name, zorder=100)

def calculate_MW_abundances(sim_info):

    select_sample = np.where((sim_info.halo_data.log10_halo_mass >= 11.95) &
                             (sim_info.halo_data.log10_halo_mass <= 12.0))[0]

    select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]

    sample = select_sample[select_centrals]
    num_sample = len(sample)

    Mg_Fe = []
    Fe_H = []
    gradius = []
    zcoord = []

    print(sim_info.halo_data.log10_stellar_mass[sample])

    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)

        mass = part_data.stars.masses.value[bound_particles_only]
        pos = part_data.stars.coordinates.value[bound_particles_only, :] * 1e3 # kpc
        vel = part_data.stars.velocities.value[bound_particles_only]

        kappa, momentum = calculate_morphology(pos, vel, mass)
        galactocentric_radius, z_coordinate = part_data.calculate_galactocentric_radius(momentum, pos)

        if kappa < 0.25: continue
        gradius = np.append(gradius, galactocentric_radius)
        zcoord = np.append(zcoord, z_coordinate)
        data = part_data.stars.calculate_abundances(("Fe_H", "Mg_Fe"))
        Mg_Fe = np.append(Mg_Fe, data["Mg_Fe"][bound_particles_only])
        Fe_H = np.append(Fe_H, data["Fe_H"][bound_particles_only])

    return Fe_H, Mg_Fe, gradius, zcoord

def plot_histogram(x, radius, z, color, label):

    bins = np.arange(-2,1,0.1)
    select_zcoord = np.where(np.abs(z)<1)[0]
    inner = np.where(radius[select_zcoord] <= 5)[0]
    weights = 1./np.ones(len(inner))
    inhist, bin_edges, _ = stat.binned_statistic(x=x[select_zcoord[inner]], values=weights, statistic="sum", bins=bins, )
    inhist /= np.max(inhist)

    middle = np.where((radius[select_zcoord] > 5) & (radius[select_zcoord] <= 9))[0]
    weights = 1./np.ones(len(middle))
    midhist, _, _ = stat.binned_statistic(x=x[select_zcoord[middle]], values=weights, statistic="sum", bins=bins, )
    midhist /= np.max(midhist)

    outer = np.where(radius[select_zcoord] > 9)[0]
    weights = 1./np.ones(len(outer))
    outhist, _, _ = stat.binned_statistic(x=x[select_zcoord[outer]], values=weights, statistic="sum", bins=bins, )
    outhist /= np.max(outhist)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
    plt.plot(bin_centers, inhist, '-', lw=1, color=color)
    plt.plot(bin_centers, midhist, '-', lw=2, color=color)
    plt.plot(bin_centers, outhist, '-', lw=3, color=color)

def plot_contours(x, y, xmin, xmax, ymin, ymax):

    ngridx = 25
    ngridy = 25

    # Create grid values first.
    xi = np.linspace(xmin, xmax, ngridx)
    yi = np.linspace(ymin, ymax, ngridy)

    # Create a histogram
    h, xedges, yedges = np.histogram2d(
        x, y, bins=(xi, yi), density=True
    )

    xbins = 0.5 * (xedges[1:] + xedges[:-1])
    ybins = 0.5 * (yedges[1:] + yedges[:-1])

    z = h.T

    # binsize = 0.2
    # grid_min = np.min(z[z>0])  # Minimum of 1 star per bin!
    grid_min = 0.1
    grid_max = np.ceil(h.max())
    binsize = (grid_max - grid_min) / 5
    levels = np.arange(grid_min, grid_max, binsize)
    # grid_min = np.log10(10)  # Minimum of 1 star per bin!
    # grid_max = np.log10(np.ceil(h.max()))
    # levels = np.arange(grid_min, grid_max, binsize)
    # levels = 10 ** levels


    print(grid_min, grid_max, binsize, levels)
    plt.contour(10**xbins, ybins, z, levels=levels, linewidths=1, cmap="plasma", zorder=1000)

def plot_scatter_MW_with_radius(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe, galactocentric_radius, z = calculate_MW_abundances(sim_info)


    color_list = ['tab:blue','tab:orange','crimson','darkblue']

    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 2.2),
        "figure.subplot.left": 0.07,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.38,
        "figure.subplot.hspace": 0.38,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    # for i in range(config_parameters.number_of_inputs):
    #
    #     sim_info = read_simulation(config_parameters, i)
    #     Fe_H, _, radius, z = calculate_MW_abundances(sim_info)
    #     plot_histogram(Fe_H, radius, z, color_list[i], sim_info.simulation_name)

    # for i in range(config_parameters.number_of_inputs):
    #
    #     sim_info = read_simulation(config_parameters, i)
    #     Fe_H, Mg_Fe, _, galactocentric_radius = calculate_MW_abundances(sim_info)
    #     # plot_median_relation(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)
    #     select = np.where(galactocentric_radius<=0.5)[0]
    #     Fe_H = Fe_H[select]
    #     Mg_Fe = Mg_Fe[select]
    #     galactocentric_radius = galactocentric_radius[select]
    #     sort = np.argsort(galactocentric_radius)
    #     im = plt.scatter(Fe_H[sort[::-1]], Mg_Fe[sort[::-1]], c=galactocentric_radius[sort[::-1]], marker='o',
    #                      cmap='jet', s=3, vmin=0, vmax=15, edgecolors='none')

    plt.plot(galactocentric_radius, Fe_H, 'o', ms=0.5)
    plot_contours(np.log10(galactocentric_radius), Fe_H, -1, 2, -2, 1)

    plt.axis([1e-1, 100, -2, 1])
    plt.xscale('log')
    # plt.axis([-2, 1, 0, 1.2])
    # xticks = np.array([-4, -3, -2, -1, 0, 1])
    # labels = ["-4","-3","-2","-1","0","1"]
    # plt.xticks(xticks, labels)

    plt.xlabel("R [kpc]")
    plt.ylabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.03,0.8],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=9, ncol=2, columnspacing=0.23)

    ####
    ax = plt.subplot(1, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plt.plot(z, Fe_H, 'o', ms=0.5)

    # for i in range(config_parameters.number_of_inputs):
    #
    #     sim_info = read_simulation(config_parameters, i)
    #     Fe_H, Mg_Fe, _, galactocentric_radius = calculate_MW_abundances(sim_info)
    #     # plot_median_relation(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)
    #     select = np.where((galactocentric_radius>0.5) & (galactocentric_radius<=1))[0]
    #     Fe_H = Fe_H[select]
    #     Mg_Fe = Mg_Fe[select]
    #     galactocentric_radius = galactocentric_radius[select]
    #     sort = np.argsort(galactocentric_radius)
    #     im = plt.scatter(Fe_H[sort[::-1]], Mg_Fe[sort[::-1]], c=galactocentric_radius[sort[::-1]], marker='o',
    #                      cmap='jet', s=3, vmin=0, vmax=15, edgecolors='none')

    # for i in range(config_parameters.number_of_inputs):
    #     sim_info = read_simulation(config_parameters, i)
    #     _, Mg_Fe, radius, z = calculate_MW_abundances(sim_info)
    #     plot_histogram(Mg_Fe, radius, z, color_list[i], sim_info.simulation_name)

    #plt.axis([-4, 1, -2, 2])
    # plt.axis([-1, 1, 0, 1.2])
    # xticks = np.array([-4, -3, -2, -1, 0, 1])
    # labels = ["-4", "-3", "-2", "-1", "0", "1"]
    # plt.xticks(xticks, labels)
    plt.axis([-15, 15, -4, 2])

    plt.xlabel("z [kpc]")
    plt.ylabel("[Fe/H]")
    # plt.ylabel("PDF")
    # plt.xlabel("[Mg/Fe]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=9, ncol=2, columnspacing=0.23)

    ####

    ax = plt.subplot(1, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    plt.plot(galactocentric_radius, Mg_Fe, 'o', ms=0.5)
    plot_contours(np.log10(galactocentric_radius), Mg_Fe, -1, 2, -1, 1)
    # for i in range(config_parameters.number_of_inputs):
    #
    #     sim_info = read_simulation(config_parameters, i)
    #     Fe_H, Mg_Fe, _, galactocentric_radius, = calculate_MW_abundances(sim_info)
    #     # plot_median_relation(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)
    #     # select = np.where(galactocentric_radius>=6)[0]
    #     select = np.where((galactocentric_radius>1) & (galactocentric_radius<=3))[0]
    #     Fe_H = Fe_H[select]
    #     Mg_Fe = Mg_Fe[select]
    #     galactocentric_radius = galactocentric_radius[select]
    #     sort = np.argsort(galactocentric_radius)
    #     im = plt.scatter(Fe_H[sort[::-1]], Mg_Fe[sort[::-1]], c=np.log10(galactocentric_radius[sort[::-1]]), marker='o',
    #                      cmap='jet', s=3, vmin=0, vmax=15, edgecolors='none')

    # plt.axis([-4, 1, -2, 2])
    # xticks = np.array([-4, -3, -2, -1, 0, 1])
    # labels = ["-4","-3","-2","-1","0","1"]
    # plt.xticks(xticks, labels)

    # plt.ylabel("[Mg/Fe]")
    # plt.xlabel("[Fe/H]")
    plt.axis([0.1, 100, -1, 1])
    plt.xscale('log')
    plt.xlabel("R [kpc]")
    plt.ylabel("[Mg/Fe]")

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.legend(loc=[0.03,0.8],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
    #            frameon=False, fontsize=9, ncol=2, columnspacing=0.23)

    # axins = inset_axes(ax, width="4%", height="100%", loc='right', borderpad=-1)
    # cb = plt.colorbar(im, cax=axins, orientation="vertical")
    # # cb.set_ticks([-2.7, -2.4, -2, -1.5, -1.1])
    # cb.set_label(r'$R_{\rm{gal}}$ [kpc]', labelpad=0.5)

    plt.savefig(config_parameters.output_directory + "MW_radial_Scatter.png", dpi=300)

def plot_stellar_distribution(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe, galactocentric_radius, z = calculate_MW_abundances(sim_info)


    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (7, 2.2),
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.97,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.38,
        "figure.subplot.hspace": 0.38,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plt.plot(galactocentric_radius, z, 'o', ms=0.5)
    #plot_contours(np.log10(galactocentric_radius), Fe_H, -1, 2, -2, 1)

    plt.axis([0.1, 100, -5, 5])
    plt.xscale('log')
    plt.xlabel("R [kpc]")
    plt.ylabel("z [kpc]")

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.03,0.8],labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=9, ncol=2, columnspacing=0.23)

    ####
    ax = plt.subplot(1, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    select = np.where(Fe_H > 0)
    plt.plot(galactocentric_radius[select], z[select], 'o', ms=0.5)

    plt.axis([0.1, 100, -5, 5])
    plt.xscale('log')
    plt.xlabel("R [kpc]")
    plt.ylabel("z [kpc]")

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=9, ncol=2, columnspacing=0.23)

    ####

    ax = plt.subplot(1, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    select = np.where(Fe_H<-0.5)
    plt.plot(galactocentric_radius[select], z[select], 'o', ms=0.5)
    # plot_contours(np.log10(galactocentric_radius), Mg_Fe, -1, 2, -1, 1)

    plt.axis([0.1, 100, -5, 5])
    plt.xscale('log')
    plt.xlabel("R [kpc]")
    plt.ylabel("z [kpc]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    plt.savefig(config_parameters.output_directory + "Stellar_distribution_"+config_parameters.name_list[0]+".png", dpi=300)

# def plot_APOGEE():
#
#     input_filename = "./plotter/Observational_data/APOGEE_data.hdf5"
#     apogee_dataset = h5py.File(input_filename, "r")
#     GalR = apogee_dataset["GalR"][:]
#     Galz = apogee_dataset["Galz"][:]
#     FE_H = apogee_dataset["FE_H"][:]
#     FE_H = FE_H + constants.Fe_over_H_Grevesse07 - constants.Fe_H_Sun_Asplund
#
#     # Plot parameters
#     params = {
#         "font.size": 11,
#         "font.family": "Times",
#         "text.usetex": True,
#         "figure.figsize": (7, 2.2),
#         "figure.subplot.left": 0.1,
#         "figure.subplot.right": 0.97,
#         "figure.subplot.bottom": 0.18,
#         "figure.subplot.top": 0.95,
#         "lines.markersize": 0.5,
#         "lines.linewidth": 0.2,
#         "figure.subplot.wspace": 0.38,
#         "figure.subplot.hspace": 0.38,
#     }
#     rcParams.update(params)
#     plt.figure()
#     ax = plt.subplot(1, 3, 1)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     plt.plot(GalR, Galz, 'o', ms=0.5)
#     # plot_contours(np.log10(galactocentric_radius), Fe_H, -1, 2, -2, 1)
#
#     plt.axis([0.1, 100, -5, 5])
#     plt.xscale('log')
#     plt.xlabel("R [kpc]")
#     plt.ylabel("z [kpc]")
#
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
#                frameon=False, fontsize=9, ncol=2, columnspacing=0.23)
#
#     ####
#     ax = plt.subplot(1, 3, 2)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     select = np.where(FE_H > 0)
#     plt.plot(GalR[select], Galz[select], 'o', ms=0.5)
#
#     plt.axis([0.1, 100, -5, 5])
#     plt.xscale('log')
#     plt.xlabel("R [kpc]")
#     plt.ylabel("z [kpc]")
#
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
#                frameon=False, fontsize=9, ncol=2, columnspacing=0.23)
#
#     ####
#
#     ax = plt.subplot(1, 3, 3)
#     plt.grid(linestyle='-', linewidth=0.3)
#
#     select = np.where(FE_H < -0.5)
#     plt.plot(GalR[select], Galz[select], 'o', ms=0.5)
#     # plot_contours(np.log10(galactocentric_radius), Mg_Fe, -1, 2, -1, 1)
#
#     plt.axis([0.1, 100, -5, 5])
#     plt.xscale('log')
#     plt.xlabel("R [kpc]")
#     plt.ylabel("z [kpc]")
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#
#     plt.savefig("APOGEE_stellar_distribution.png",dpi=300)

def read_MW_abundances(sim_info, element_list):

    filename = "./outputs/MWGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
    with h5py.File(filename, "r") as file:
        x = file["Data/"+element_list[0]][:]
        y = file["Data/"+element_list[1]][:]
        R = file["Data/GalactocentricRadius"][:]
        z = file["Data/GalactocentricZ"][:]

        select = np.where((R<100) & (np.abs(z)<10))[0]

    return x[select], y[select]

def plot_abundance_ratio_MgFe(config_parameters):

    color_list = ['tab:blue', 'tab:orange', 'crimson', 'darkblue']

    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (5, 2.5),
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.97,
        "figure.subplot.bottom": 0.17,
        "figure.subplot.top": 0.95,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_MW_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        plot_median_relation(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)

    plot_APOGEE(("Fe_H", "Mg_Fe"))
    plt.axis([-4, 1, -2, 2])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)

    plt.ylabel("[Mg/Fe]")
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.legend(loc=[0.03, 0.8], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
               frameon=False, fontsize=9, ncol=2, columnspacing=0.5)

    #####
    ax = plt.subplot(1, 2, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, O_Fe = read_MW_abundances(sim_info, ("Fe_H", "O_Fe"))
        plot_median_relation(Fe_H, O_Fe, color_list[i], sim_info.simulation_name)

    plot_APOGEE(("Fe_H", "O_Fe"))
    plt.axis([-4, 1, -2, 2])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)
    plt.ylabel("[O/Fe]")
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    plt.savefig(config_parameters.output_directory + "Scatter_abundance_ratio_MgFe.png", dpi=300)


def plot_scatter_of_MW(config_parameters):

    color_list = ['steelblue', 'lightskyblue', 'y', 'salmon']

    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (5.5, 2.5),
        "figure.subplot.left": 0.11,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.top": 0.85,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_MW_data('Mg')
    plot_Pristine()
    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_MW_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        plot_median_relation_one_sigma(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name)

    plt.axis([-4, 1, -1.5, 1])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)

    plt.ylabel("[Mg/Fe]")
    plt.xlabel("[Fe/H]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [5, 9, 13, 17]],
                         ["No diffusion", "Low diffusion", "Default diffusion", "High diffusion"],
                         loc=[0.0, 1.08],
                         labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
                         frameon=False, fontsize=9, ncol=4, columnspacing=1.2)
    ax.add_artist(legend1)

    legend2 = plt.legend([lines[i] for i in [0, 1]],
                         ["MW-data compilation", "Pristine Survey"],
                         loc=[0.0, 1.01], labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
                         frameon=False, fontsize=9, ncol=2, columnspacing=1.2)
    ax.add_artist(legend2)

    #########################
    ax = plt.subplot(1, 2, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_Pristine_scatter()
    plot_MW_scatter()
    plot_APOGEE_scatter()
    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        Fe_H, Mg_Fe = read_MW_abundances(sim_info, ("Fe_H", "Mg_Fe"))
        plot_scatter(Fe_H, Mg_Fe, color_list[i], sim_info.simulation_name,'-')

        # sim_info = read_simulation(config_parameters, i)
        # Fe_H, Mg_Fe, sig_Fe, sig_Mg = read_MW_scatter_in_abundances(sim_info,("Fe_H","Mg_Fe"))
        # _, O_Fe, _, sig_O = read_MW_scatter_in_abundances(sim_info,("Fe_H","O_Fe"))
        #
        # plot_error_bar(1 + delta, sig_Mg, color_list[i], None)
        # plot_error_bar(2 + delta, sig_O, color_list[i], None)
        # delta += 0.01

    # plot_APOGEE_scatter(("Fe_H","Mg_Fe"),True)
    # plot_APOGEE_scatter(("Fe_H","O_Fe"), False)
    # plot_MW_scatter("Mg",True)
    # plot_MW_scatter("O",False)

    plt.axis([-4, 1, 0, 1])
    xticks = np.array([-4, -3, -2, -1, 0, 1])
    labels = ["$-4$", "$-3$", "$-2$", "$-1$", "$0$", "$1$"]
    plt.xticks(xticks, labels)
    plt.xlabel("[Fe/H]")
    plt.ylabel("Scatter [Mg/Fe] (84th-16th)")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [2]],
                         ["APOGEE survey"],
                         loc=[-0.15, 1.01],
                         labelspacing=0.05, handlelength=0.5, handletextpad=0.3,
                         frameon=False, fontsize=9, ncol=4, columnspacing=1.2)
    ax.add_artist(legend1)


    plt.savefig(config_parameters.output_directory + "Scatter_MW.png", dpi=300)
