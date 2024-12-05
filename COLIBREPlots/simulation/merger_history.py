import numpy as np
import h5py
from tqdm import tqdm
from simulation import particle_data
from simulation.simulation_data import read_simulation

def make_merger_tree(config_parameters):

    # Reading z=2 snapshot info
    sim_info_z2 = read_simulation(config_parameters, 1)
    particles_file = h5py.File(f"{sim_info_z2.directory}/{sim_info_z2.catalogue_particles}", "r")
    group_file = h5py.File(f"{sim_info_z2.directory}/{sim_info_z2.catalogue_groups}", "r")
    halo_group_file_z2 = group_file["Offset"]
    particle_ids_in_file_z2 = particles_file["Particle_IDs"]

    # Creating z=1 snapshot halo sample
    sim_info_z1 = read_simulation(config_parameters, 0)
    select_sample = np.where(sim_info_z1.halo_data.log10_stellar_mass >= 8.0)[0]
    select_centrals = np.where(sim_info_z1.halo_data.type[select_sample] == 10)[0]

    sample = select_sample[select_centrals]
    num_sample = len(sample)

    halo_indx_z1 = sim_info_z1.halo_data.halo_index[sample]
    halo_indx_z2 = np.zeros(num_sample)
    num_progenitor = np.zeros(num_sample)

    for i in tqdm(range(num_sample)):

        part_data = particle_data.load_particle_data(sim_info_z1, sample[i])
        pos = part_data.dm.coordinates.value[:, :] * 1e3  # kpc

        radius = np.linalg.norm(pos[:, :3], axis=1)
        sorting = np.argsort(radius)
        select_closest = sorting[:min(10000, len(sorting))]

        ids_z1 = part_data.dm.ids[select_closest]
        _, _, indx_ids_z2 = np.intersect1d(ids_z1, particle_ids_in_file_z2,
                                           assume_unique=True, return_indices=True, )

        hist, _ = np.histogram(indx_ids_z2, bins = halo_group_file_z2)
        all_halo_indx_z2 = np.flatnonzero(hist > 0)
        main_indx = np.argmax(hist)
        halo_indx_z2[i] = main_indx
        num_progenitor[i] = len(all_halo_indx_z2)
        print(halo_indx_z1[i], halo_indx_z2[i], len(all_halo_indx_z2))


    galaxy_stellar_mass = sim_info_z1.halo_data.log10_stellar_mass[sample]
    halo_mass = sim_info_z1.halo_data.log10_halo_mass[sample]

    # Output data
    output_file = f"{sim_info_z1.output_path}/Merger_history_" + sim_info_z1.simulation_name + ".hdf5"
    data_file = h5py.File(output_file, 'w')
    f = data_file.create_group('Data')
    f.create_dataset('HaloID', data=halo_indx_z1)
    f.create_dataset('ProgenitorID', data=halo_indx_z2.astype('int'))
    f.create_dataset('NumProgenitors', data=num_progenitor)
    f.create_dataset('Stellar_mass', data=galaxy_stellar_mass)
    f.create_dataset('Halo_mass', data=halo_mass)
    data_file.close()


