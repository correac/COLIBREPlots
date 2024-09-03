import numpy as np
import unyt
from unyt import unyt_array
from unyt import Mpc, kpc, pc, Msun

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio import load, mask

#from swiftascmaps import evermore, evermore_r, red_tv, red_tv_r, reputation, midnights
from astropy.visualization import make_lupton_rgb

def get_projection_map(model, npix, size, rotation):

    origin = model.metadata.boxsize
    size = unyt.unyt_array([size, size, size], 'kpc')
    region = [[-1.0 * b + 0.5 * o, b + 0.5 * o] for b, o in zip(size, origin)]

    region_defined = unyt.unyt_array([region[0][0], region[0][1], region[1][0], region[1][1]], 'kpc')

    # Calculate particles luminosity in i, r and g bands
    luminosities = [
        model.stars.luminosities.GAMA_i,
        model.stars.luminosities.GAMA_r,
        model.stars.luminosities.GAMA_g,
    ]

    rgb_image_face = np.zeros((npix, npix, len(luminosities)))

    for ilum in range(len(luminosities)):

        model.stars.usermass = luminosities[ilum]

        if rotation == 'False':
            pixel_grid = project_pixel_grid(
                data=model.stars,
                resolution=npix,
                project="usermass",
                parallel=True,
                region=region_defined,
                boxsize=model.metadata.boxsize,
                backend="subsampled",
            )
        else:

            ang_momentum = np.array([0, 0, 1])
            edge_on_rotation_matrix = rotation_matrix_from_vector(ang_momentum, axis="y")

            pixel_grid = project_pixel_grid(
                data=model.stars,
                resolution=npix,
                project="usermass",
                parallel=True,
                region=region_defined,
                rotation_center=origin,
                rotation_matrix=edge_on_rotation_matrix,
                boxsize=model.metadata.boxsize,
                backend="subsampled",
            )

        x_range = region[0][1] - region[0][0]
        y_range = region[1][1] - region[1][0]
        units = 1.0 / (x_range * y_range)
        units.convert_to_units(1.0 / (x_range.units * y_range.units))

        mass_map_face = unyt_array(pixel_grid, units=units)
        mass_map_face.convert_to_units(1.0 / pc ** 2)
        rgb_image_face[:, :, ilum] = mass_map_face.T


    Q=10
    stretch=0.1

    stars_map = make_lupton_rgb(
        rgb_image_face[:, :, 0],
        rgb_image_face[:, :, 1],
        rgb_image_face[:, :, 2],
        Q=Q,
        stretch=stretch,
    )

    return stars_map, region_defined

def make_image(sim_info, npix, size):

    stars_map, region = get_projection_map(sim_info, npix, size)

    density = np.log10(sim_info.gas.densities.to("Msun/kpc**3"))
    arg_sort = np.argsort(density)
    density = density[arg_sort]

    pos_x = sim_info.gas.coordinates[arg_sort,0] - sim_info.metadata.boxsize[0] * 0.5
    pos_y = sim_info.gas.coordinates[arg_sort,1] - sim_info.metadata.boxsize[0] * 0.5

    denmin = np.min(density)
    denmax = np.max(density)

    ######################
    fig = plt.figure(figsize=(6.0, 6.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = plt.subplot(1,1,1)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.imshow(stars_map, extent=region, interpolation='nearest')

    filename = "stars_light_map_galaxy"
    filename += "_size_%.2dkpc"%size
    fig.savefig(filename+".png", dpi=500)
    plt.close()

    # Nienke select here a colormap
    #colormap = evermore_r
    # colormap = midnights
    # colormap = 'binary_r'
    # colormap = 'viridis'
    #colormap = 'magma'


    fig = plt.figure(figsize=(6.0, 6.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = plt.subplot(1,1,1)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_axis_off()

    plt.scatter(pos_x, pos_y, c=density, s=10, vmin=denmin, vmax=denmax, cmap="magma", edgecolors="none",)
    # ax.autoscale(False)

    # ax.imshow(LogNorm()(gas_map.value), extent=region, interpolation='nearest', cmap=colormap)
    filename = "gas_mass_map_galaxy_"
    filename += "_size_%dkpc"%size
    fig.savefig(filename+".png", dpi=500)
    plt.close()


def stars_light_map_face_on(ax, file_path, filename):

    filename = file_path + filename
    model = load(filename)

    # Some options:
    npix = 1024         # number of pixels
    size = 40          # [kpc] size of image

    stars_map, region = get_projection_map(model, npix, size, rotation='False')
    ax.imshow(stars_map, extent=region, interpolation='nearest')

    ax.invert_xaxis()
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.text(0.15, 0.05, 'Stars', horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='white')


def stars_light_map_edge_on(ax, file_path, filename):

    filename = file_path + filename
    model = load(filename)

    # Some options:
    npix = 1024         # number of pixels
    size = 40          # [kpc] size of image

    stars_map, region = get_projection_map(model, npix, size, rotation='True')
    ax.imshow(stars_map, extent=region, interpolation='nearest')

    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.text(0.15, 0.05, 'Stars', horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='white')

def gas_map_face_on(ax, file_path, filename):

    # file_path = "/Users/cc276407/Simulation_data/cosma/IsolatedGalaxy/IsolatedGalaxy_randomZ/"
    # filename = file_path + "output_00%02d.hdf5" % 18
    filename = file_path + filename

    model = load(filename)

    density = np.log10(model.gas.densities.to("Msun/kpc**3"))
    arg_sort = np.argsort(density)
    density = density[arg_sort]

    pos_x = model.gas.coordinates[arg_sort,0] - model.metadata.boxsize[0] * 0.5
    pos_y = model.gas.coordinates[arg_sort,1] - model.metadata.boxsize[0] * 0.5

    # denmin = np.min(density)
    # denmax = np.max(density)
    # print(denmin, denmax)

    im = plt.scatter(pos_x, pos_y, c=density, s=10, vmin=4, vmax=10, cmap="magma", edgecolors="none",)

    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.text(0.1, 0.05, 'Gas', horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, color='black')

    return im

