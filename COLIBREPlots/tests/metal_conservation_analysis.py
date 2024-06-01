import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
import numpy as np
import h5py
from galaxy_visualization_plot import stars_light_map_face_on, gas_map_face_on, stars_light_map_edge_on

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

def metallicity_distribution_plot(ax, file_path, filename):

    Z_all, time, x_pos, y_pos, Z = read_data(file_path, filename)

    cmap = plt.cm.twilight
    normalize = matplotlib.colors.TwoSlopeNorm(vmin=0.8, vcenter=1, vmax=1.2)
    im = ax.scatter(x_pos, y_pos, s=1, c=Z, cmap=cmap, edgecolors=None, norm=normalize)

    ax.axis([-30, 30, -30, 30])
    ax.set_xlabel("x [kpc]", labelpad=0)
    ax.set_ylabel("y [kpc]", labelpad=0)
    ax.text(0.1, 0.05, 'Gas', horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='black')

    # cbar = plt.colorbar(im, label='$Z/Z_{\odot}$', fraction=0.05, pad=0.03, extend='min')
    # cbar.ax.set_yticks([0.8,1,1.2])


    return im


def total_metal_evolution(ax, file_path, snap):

    if snap > 0:
        Z_track = np.zeros(snap)
        time_track = np.zeros(snap)
        for i in range(0, snap):
            filename = "output_00%02d.hdf5" % i
            Z_all, time, _, _, _ = read_data(file_path, filename)
            Z_track[i] = Z_all
            time_track[i] = time

        Delta_Z = (Z_track - Z_track[0]) / Z_track[0]

    else:
        time_track = np.zeros(1)
        Delta_Z = np.zeros(1)

    ax.plot(time_track, Delta_Z / 1e-5, '-o', color='darkblue')
    ax.set_xlabel("Time [Myr]", labelpad=0)
    ax.set_ylabel(r"$(Z-Z(t=0))/Z(t=0)$ [$\times 10^{-5}$]", labelpad=0)
    ax.set_xlim(0,500)


if __name__ == "__main__":

    initial_snap = 0
    for i in range(initial_snap, 18):

        file_path = "/Users/cc276407/Simulation_data/cosma/IsolatedGalaxy/IsolatedGalaxy_randomZ/"
        filename = "output_00%02d.hdf5" % i

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
        # cbar.ax.set_yticks([0.8, 1, 1.2])

        #####
        ax0 = plt.subplot(3, 3, 4)
        total_metal_evolution(ax0, file_path, i)

        #####
        ax1 = plt.subplot(3, 3, 5)
        _ = metallicity_distribution_plot(ax1, file_path, "output_0000.hdf5")

        #####
        ax2 = plt.subplot(3, 3, 6)
        im = metallicity_distribution_plot(ax2, file_path, filename)
        axlist = [ax0, ax1, ax2]
        cbar = fig.colorbar(im, ax=axlist, label='$Z/Z_{\odot}$', fraction=0.05, pad=0.03, extend='min')
        cbar.ax.set_yticks([0.8, 1, 1.2])

        #####
        ax0 = plt.subplot(3, 3, 7)
        file_path = "/Users/cc276407/Simulation_data/cosma/IsolatedGalaxy/IsolatedGalaxy_halfbox/"
        total_metal_evolution(ax0, file_path, i)

        #####
        ax1 = plt.subplot(3, 3, 8)
        _ = metallicity_distribution_plot(ax1, file_path, "output_0000.hdf5")

        #####
        ax2 = plt.subplot(3, 3, 9)
        im = metallicity_distribution_plot(ax2, file_path, filename)

        axlist = [ax0, ax1, ax2]
        cbar = fig.colorbar(im, ax=axlist, label='$Z/Z_{\odot}$', fraction=0.05, pad=0.03, extend='min')
        cbar.ax.set_yticks([0.8,1,1.2])

        plt.savefig("metal_conservation_snap_00%02d.png"%i, dpi=300)

