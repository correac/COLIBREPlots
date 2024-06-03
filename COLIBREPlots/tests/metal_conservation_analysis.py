import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from pylab import rcParams
import numpy as np
import h5py
from galaxy_visualization_plot import stars_light_map_face_on, gas_map_face_on, stars_light_map_edge_on

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def read_data(file_path, filename):

    filename = file_path+filename
    Zsun = 0.0133714

    with h5py.File(filename, "r") as file:
        part_data_group = file["PartType0"]
        pos = part_data_group["Coordinates"][:,:]
        Z = part_data_group["MetalMassFractions"][:]
        boxSize = file["/Header"].attrs["BoxSize"][0]
        timeSU = file["/Header"].attrs["Time"][:]
        unit_time_in_cgs = file["/Units"].attrs["Unit time in cgs (U_t)"]

    Z_all = np.sum(Z)
    year_in_cgs = 3600.0 * 24 * 365.0
    t_Myr = timeSU * unit_time_in_cgs / year_in_cgs / 1.0e6
    time = t_Myr

    x = pos[:, 0] - boxSize / 2
    y = pos[:, 1] - boxSize / 2
    z_coord = pos[:, 2] - boxSize / 2

    select = np.where(np.abs(z_coord)<0.1)[0]
    x = x[select]
    y = y[select]
    Z = Z[select] / Zsun

    return Z_all, time, x, y, Z

def metallicity_distribution_plot(ax, file_path, filename, label):

    Z_all, time, x_pos, y_pos, Z = read_data(file_path, filename)

    cmap = plt.cm.twilight
    new_cmap = truncate_colormap(cmap, 0.0, 0.8)

    normalize = matplotlib.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.6, vmax=1.1)
    im = ax.scatter(x_pos, y_pos, s=1, c=Z, cmap=new_cmap, edgecolors=None, norm=normalize)

    ax.axis([-30, 30, -30, 30])
    ax.set_xlabel("x [kpc]", labelpad=0)
    ax.set_ylabel("y [kpc]", labelpad=0)
    ax.text(0.05, 0.95, label, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))

    ax.text(0.02, 0.02, 'Gas', horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, color='black')

    return im


def total_metal_evolution(ax, file_path, snap, color, label):

    if snap > 0:
        Z_track = np.zeros(snap)
        time_track = np.zeros(snap)
        for i in range(0, snap):
            filename = "output_0%003d.hdf5" % i
            Z_all, time, _, _, _ = read_data(file_path, filename)
            Z_track[i] = Z_all
            time_track[i] = time / 1e3

        Delta_Z = (Z_track - Z_track[0]) / Z_track[0]

    else:
        time_track = np.zeros(1)
        Delta_Z = np.zeros(1)

    ax.plot(time_track, Delta_Z / 1e-5, '-o',lw=1, color=color, label=label)
    ax.set_xlabel("Time [Gyr]", labelpad=0)
    ax.set_ylabel(r"$(Z-Z(t=0))/Z(t=0)$ [$\times 10^{-5}$]", labelpad=0)
    ax.set_xlim(0,2)


if __name__ == "__main__":

    initial_snap = 0
    for i in range(initial_snap, 1):

        file_path = "/Users/cc276407/Simulation_data/cosma/IsolatedGalaxy/IsolatedGalaxy_halfbox/"
        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_randomZ/"
        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_halfboxZ_nocooling/"
        filename = "output_0%003d.hdf5" % i

        # Plot parameters
        params = {
            "font.size": 12,
            "font.family": "Times",
            "text.usetex": True,
            "figure.figsize": (9.2, 8),
            "figure.subplot.left": 0.08,
            "figure.subplot.right": 0.94,
            "figure.subplot.bottom": 0.08,
            "figure.subplot.top": 0.98,
            "lines.markersize": 0.5,
            "lines.linewidth": 0.2,
            "figure.subplot.wspace": 0.3,
            "figure.subplot.hspace": 0.2,
        }
        rcParams.update(params)
        fig = plt.figure()

        #####
        ax0 = plt.subplot(3, 3, 1)
        stars_light_map_face_on(ax0, file_path, filename)

        #####
        ax1 = plt.subplot(3, 3, 2)
        stars_light_map_edge_on(ax1, file_path, filename)

        #####
        ax2 = plt.subplot(3, 3, 3)
        im = gas_map_face_on(ax2, file_path, filename)
        axlist = [ax0, ax1, ax2]
        cbar = fig.colorbar(im, ax=axlist, label='Gas Density [M$_{\odot}$ kpc$^{-3}$]', fraction=0.05, pad=0.03, extend='min')
        cbar.ax.set_yticks([4, 6, 8, 10])

        #####
        ax0 = plt.subplot(3, 3, 4)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_halfboxZ_nocooling/"
        # filename = "output_0%003d.hdf5" % i
        total_metal_evolution(ax0, file_path, i, 'darkblue', '$C_{\mathrm{diff}}=0.01$')

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_halfboxZ_nocooling_C1/"
        # filename = "output_0%003d.hdf5" % i
        total_metal_evolution(ax0, file_path, i, 'salmon', '$C_{\mathrm{diff}}=1$')

        plt.legend(loc=[0.1, 0.85], labelspacing=0.05, handlelength=0.7, handletextpad=0.1,
                   frameon=False, fontsize=12, ncol=2, columnspacing=0.5)

        #####
        ax1 = plt.subplot(3, 3, 5)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_halfboxZ_nocooling/"
        # filename = "output_0%003d.hdf5" % i

        _ = metallicity_distribution_plot(ax1, file_path, filename, '$C_{\mathrm{diff}}=0.01$')

        #####
        ax2 = plt.subplot(3, 3, 6)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_halfboxZ_nocooling_C1/"
        # filename = "output_0%003d.hdf5" % i

        im = metallicity_distribution_plot(ax2, file_path, filename, '$C_{\mathrm{diff}}=1$')
        axlist = [ax0, ax1, ax2]
        cbar = fig.colorbar(im, ax=axlist, label='$Z/Z_{\odot}$', fraction=0.05, pad=0.03)
        cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.1])

        #####
        ax0 = plt.subplot(3, 3, 7)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_randomZ_nocooling/"
        # filename = "output_0%003d.hdf5" % i
        total_metal_evolution(ax0, file_path, i, 'darkblue', '$C_{\mathrm{diff}}=0.01$')

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_randomZ_nocooling_C1/"
        # filename = "output_0%003d.hdf5" % i
        total_metal_evolution(ax0, file_path, i, 'salmon', '$C_{\mathrm{diff}}=1$')

        plt.legend(loc=[0.1, 0.85], labelspacing=0.05, handlelength=0.7, handletextpad=0.1,
                   frameon=False, fontsize=12, ncol=2, columnspacing=0.5)
        #####
        ax1 = plt.subplot(3, 3, 8)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_randomZ_nocooling/"
        # filename = "output_0%003d.hdf5" % i
        _ = metallicity_distribution_plot(ax1, file_path, filename, '$C_{\mathrm{diff}}=0.01$')

        #####
        ax2 = plt.subplot(3, 3, 9)

        # file_path = "/cosma7/data/dp004/dc-corr1/SIMULATION_RUNS/COLIBRE_05_2024/IsolatedGalaxy/IsolatedGalaxy_randomZ_nocooling_C1/"
        # filename = "output_0%003d.hdf5" % i

        im = metallicity_distribution_plot(ax2, file_path, filename, '$C_{\mathrm{diff}}=1$')
        axlist = [ax0, ax1, ax2]
        cbar = fig.colorbar(im, ax=axlist, label='$Z/Z_{\odot}$', fraction=0.05, pad=0.03)
        cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.1])
        plt.savefig("metal_conservation_snap_0%003d.png"%i, dpi=300)

