import numpy as np
import h5py
from tqdm import tqdm
from simulation import particle_data
from simulation.simulation_data import read_simulation
from astropy.cosmology import Planck13 as cosmo
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

def read_merger_history(sim_info):

    file = f"{sim_info.output_path}/Merger_history_" + sim_info.simulation_name + ".hdf5"
    data_file = h5py.File(file, 'r')
    halo_indx_z1 = data_file["Data/HaloID"]
    halo_indx_z2 = data_file["Data/ProgenitorID"]
    return halo_indx_z1, halo_indx_z2.astype('int')

def read_mask(sim_info, halo_indx_z):

    halo_indx = sim_info.halo_data.halo_index

    _, match_in_halo_indx, match_in_z = np.intersect1d(halo_indx, halo_indx_z, assume_unique=True, return_indices=True, )

    mask = np.ones(len(halo_indx_z)) * (-1)
    mask[match_in_z] = match_in_halo_indx

    return mask.astype('int')

def calculate_outflow_rates(config_parameters):

    # Creating z=1 snapshot halo sample
    sim_info_z1 = read_simulation(config_parameters, 0)
    sim_info_z2 = read_simulation(config_parameters, 1)
    halo_indx_z1, halo_indx_z2 = read_merger_history(sim_info_z1)
    num_sample = len(halo_indx_z1)

    mask_z1 = read_mask(sim_info_z1, halo_indx_z1)
    mask_z2 = read_mask(sim_info_z2, halo_indx_z2)

    sfr = sim_info_z1.halo_data.sfr[mask_z1] # Msun/yr
    R200 = sim_info_z1.halo_data.virial_radius[mask_z1]
    Vmax = sim_info_z1.halo_data.vmax[mask_z1]

    outflow_rate = np.zeros(num_sample)
    metal_outflow_rate = np.zeros(num_sample)
    loading_factor = np.zeros(num_sample)
    metal_loading_factor = np.zeros(num_sample)

    accretion_rate = np.zeros(num_sample)
    metal_accretion_rate = np.zeros(num_sample)

    delta_time = cosmo.age(0).value - cosmo.age(0.1).value # Gyr
    delta_time *= 1e9 # yr
    delta_time_sec = delta_time * 365.2425 * 24 * 3600  # sec

    # for i in tqdm(range(num_sample)):
    for i in tqdm(range(200)):

        # Patch flag:
        if ((mask_z1[i] == -1) or (mask_z2[i] == -1)): continue

        # z=0 #########
        part_data = particle_data.load_particle_data(sim_info_z1, mask_z1[i])
        bound_particles_only = part_data.select_bound_particles(sim_info_z1, halo_indx_z1[i], part_data.gas.ids)

        if len(bound_particles_only) < 10: continue # low number statistics..

        T = part_data.gas.temperature[bound_particles_only]
        nH = part_data.gas.hydrogen_number_density[bound_particles_only].value
        pos = part_data.gas.coordinates.value[bound_particles_only, :] * 1e3  # kpc
        radius = np.linalg.norm(pos[:, :3], axis=1)

        select_cold_dense = np.where( (T<10**4.5) & (nH>0.1) & (radius <= 0.25 * R200[i]))[0]

        if len(select_cold_dense) < 10: continue

        ids_z1 = part_data.gas.ids[bound_particles_only[select_cold_dense]]
        mass_z1 = part_data.gas.masses.value[bound_particles_only[select_cold_dense]]
        metallicity_z1 = part_data.gas.metal_mass_fractions[bound_particles_only[select_cold_dense]]


        # z=0.1 #########
        part_data = particle_data.load_particle_data(sim_info_z2, mask_z2[i])
        bound_particles_only = part_data.select_bound_particles(sim_info_z2, halo_indx_z2[i], part_data.gas.ids)

        if len(bound_particles_only) < 10: continue
        T = part_data.gas.temperature[bound_particles_only]
        nH = part_data.gas.hydrogen_number_density[bound_particles_only].value
        pos = part_data.gas.coordinates.value[bound_particles_only, :] * 1e3  # kpc
        radius = np.linalg.norm(pos[:, :3], axis=1)

        select_cold_dense = np.where( (T<10**4.5) & (nH>0.1) & (radius <= 0.25 * R200[i]))[0]

        if len(select_cold_dense) < 10: continue

        ids_z2 = part_data.gas.ids[bound_particles_only[select_cold_dense]]
        mass = part_data.gas.masses.value[bound_particles_only[select_cold_dense]]
        metallicity = part_data.gas.metal_mass_fractions[bound_particles_only[select_cold_dense]]
        radius = radius[select_cold_dense]

        _, indx_ids_z1, indx_ids_z2 = np.intersect1d(ids_z1, ids_z2, assume_unique=True, return_indices=True, )

        # We calculate inflowing gas rates:
        sub_indx_ids_z1 = np.delete(np.arange(len(ids_z1)), indx_ids_z1) # particles not in match (new)
        accretion_rate[i] = np.sum(mass_z1[sub_indx_ids_z1])
        metal_accretion_rate[i] = np.sum(metallicity_z1[sub_indx_ids_z1] * mass_z1[sub_indx_ids_z1])
        accretion_rate[i] /= delta_time  # Msun / yr
        metal_accretion_rate[i] /= delta_time  # Msun / yr

        # Now we calculate conditions for selecting outflowing gas:
        # We follow Mitchel+ (2020)
        sub_indx_ids_z2 = np.delete(np.arange(len(ids_z2)), indx_ids_z2) # particles not in match (lost)
        part_data = particle_data.load_particle_data(sim_info_z1, mask_z1[i])
        ids_z1 = part_data.gas.ids
        _, indx_ids_z1, indx_ids_z2b = np.intersect1d(ids_z1, ids_z2[sub_indx_ids_z2], assume_unique=True, return_indices=True, )

        pos = part_data.gas.coordinates.value[indx_ids_z1, :] * 1e3  # kpc
        radius_z1 = np.linalg.norm(pos[:, :3], axis=1)

        condition_1_for_outflowing = (radius_z1 - radius[sub_indx_ids_z2[indx_ids_z2b]]) * 3.086e16 / delta_time_sec # km / sec
        condition_1_for_outflowing -= 0.25 * Vmax[i]

        vel = part_data.gas.velocities.value[indx_ids_z1, :]
        vrad = np.linalg.norm(vel[:, :3], axis=1)

        condition_2_for_outflowing = vrad - 0.125 * Vmax[i]

        select_outflowing = np.where((condition_1_for_outflowing > 0) & (condition_2_for_outflowing > 0))[0]

        outflow_rate[i] = np.sum(mass[sub_indx_ids_z2[indx_ids_z2b[select_outflowing]]])
        metal_outflow_rate[i] = np.sum(metallicity[sub_indx_ids_z2[indx_ids_z2b[select_outflowing]]] * mass[sub_indx_ids_z2[indx_ids_z2b[select_outflowing]]])
        outflow_rate[i] /= delta_time # Msun / yr
        metal_outflow_rate[i] /= delta_time # Msun / yr

        loading_factor[i] = outflow_rate[i] / sfr[i]
        metal_loading_factor[i] = metal_outflow_rate[i] / sfr[i]


    galaxy_stellar_mass = sim_info_z1.halo_data.log10_stellar_mass[mask_z1]
    halo_mass = sim_info_z1.halo_data.log10_halo_mass[mask_z1]

    # Output data
    output_file = f"{sim_info_z1.output_path}/Outflow_rates_" + sim_info_z1.simulation_name + ".hdf5"
    data_file = h5py.File(output_file, 'w')
    f = data_file.create_group('Data')
    f.create_dataset('HaloID', data=halo_indx_z1)
    f.create_dataset('ProgenitorID', data=halo_indx_z2)
    f.create_dataset('Stellar_mass', data=galaxy_stellar_mass)
    f.create_dataset('Halo_mass', data=halo_mass)
    f.create_dataset('SFR', data=sfr)

    galaxy_stellar_mass = sim_info_z2.halo_data.log10_stellar_mass[mask_z2]
    halo_mass = sim_info_z2.halo_data.log10_halo_mass[mask_z2]
    f.create_dataset('Progenitor_Stellar_mass', data=galaxy_stellar_mass)
    f.create_dataset('Progenitor_Halo_mass', data=halo_mass)

    f.create_dataset('Outflow_rate', data=outflow_rate)
    f.create_dataset('Metal_outflow_rate', data=metal_outflow_rate)

    f.create_dataset('Loading_factor', data=loading_factor)
    f.create_dataset('Metal_loading_factor', data=metal_loading_factor)

    f.create_dataset('Accretion_rate', data=accretion_rate)
    f.create_dataset('Metal_accretion_rate', data=metal_accretion_rate)

    data_file.close()


def read_rates(sim_info):

    file = f"{sim_info.output_path}/Outflow_rates_" + sim_info.simulation_name + ".hdf5"
    data_file = h5py.File(file, 'r')
    outflow_rate = data_file["Data/Outflow_rate"][:]
    metal_outflow_rate = data_file["Data/Metal_outflow_rate"][:]
    accretion_rate = data_file["Data/Accretion_rate"][:]
    metal_accretion_rate = data_file["Data/Metal_accretion_rate"][:]
    loading_factor = data_file["Data/Loading_factor"][:]
    metal_loading_factor = data_file["Data/Metal_loading_factor"][:]
    stellar_mass = data_file["Data/Stellar_mass"][:]
    halo_mass = data_file["Data/Halo_mass"][:]
    sfr = data_file["Data/SFR"][:]

    # select = np.where(outflow_rate > 0)[0]

    select = np.where((outflow_rate > 0) & (sfr > 0))[0]

    return {"outflow_rate":outflow_rate[select], "metal_outflow_rate":metal_outflow_rate[select],
            "accretion_rate":accretion_rate[select], "metal_accretion_rate":metal_accretion_rate[select],
            "loading_factor":loading_factor[select], "metal_loading_factor":metal_loading_factor[select],
            "stellar_mass":stellar_mass[select], "halo_mass":halo_mass[select]}


    return halo_indx_z1, halo_indx_z2.astype('int')


def plot_median_relation_one_sigma(x, y, color, output_name):

    num_min_per_bin = 5
    bins = np.arange(7, 13, 0.25)
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


def plot_rates(config_parameters):

    color_list = ['steelblue', 'lightskyblue', 'y', 'salmon']
    names = ['No diffusion', 'Low diffusion', 'Default diffusion', 'High diffusion']

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
        "figure.subplot.wspace": 0.35,
        "figure.subplot.hspace": 0.35,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(2, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["outflow_rate"], color_list[i], names[i])

    plt.axis([8, 11, 0.01, 10])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\dot{M}_{\mathrm{out}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    plt.legend(loc=[0, 0.6], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=10, ncol=1, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["metal_outflow_rate"], color_list[i], sim_info.simulation_name)

    plt.axis([8, 11, 1e-4, 1])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\dot{M}_{\mathrm{out,Z}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["loading_factor"], color_list[i], sim_info.simulation_name)

    plt.axis([8, 11, 0.1, 100])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\eta_{\mathrm{out}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)


    #######
    ax = plt.subplot(2, 3, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["metal_loading_factor"], color_list[i], sim_info.simulation_name)

    plt.axis([8, 11, 0.001, 5])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\eta_{\mathrm{out,Z}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 3, 5)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["accretion_rate"], color_list[i],
                                       sim_info.simulation_name)

    plt.axis([8, 11, 0.01, 10])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\dot{M}_{\mathrm{in}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 3, 6)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_rates(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["metal_accretion_rate"], color_list[i],
                                       sim_info.simulation_name)

    plt.axis([8, 11, 0.0001, 1])
    plt.yscale('log')
    # plt.xticks([-2, -1, 0, 0.5])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$\eta_{\mathrm{in,Z}}$ [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    plt.savefig(config_parameters.output_directory + "outflow_rates.png", dpi=300)