import numpy as np
from tqdm import tqdm
import h5py
from simulation import particle_data
from simulation.simulation_data import read_simulation
from simulation.satellites_catalogue import look_for_satellites


def calculate_galaxy_satellite_abundances(sim_info):

    sample_hosts = np.where((sim_info.halo_data.log10_halo_mass >= 11.8) &
                             (sim_info.halo_data.log10_halo_mass <= 12.2))[0]

    select_centrals = np.where(sim_info.halo_data.type[sample_hosts] == 10)[0]

    sample_hosts = sample_hosts[select_centrals] # Here I am selecting MW-type haloes
    sample_satellites, index_satellites, index_host = look_for_satellites(sim_info, sample_hosts)
    num_sample = len(sample_hosts)

    C_Fe = []
    N_Fe = []
    Mg_Fe = []
    O_Fe = []
    Si_Fe = []
    Ne_Fe = []
    Fe_H = []

    host_halo_indx = []
    host_subhalo_indx = []
    host_galaxy_stellar_mass = []

    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample_hosts[i]]
        select_satellites = np.where(index_host == halo_indx)[0]
        if len(select_satellites) == 0 : continue  # No satellites found for this halo

        Ms = sim_info.halo_data.log10_stellar_mass[sample_satellites[select_satellites]] # Satellites stellar mass
        sort_Ms = np.argsort(Ms)  # Ordering satellites according to their mass
        Ms = Ms[sort_Ms]
        sample_satellites_i = sample_satellites[select_satellites[sort_Ms]]
        index_satellites_i = index_satellites[select_satellites[sort_Ms]]

        num_satellites = len(select_satellites)

        for j in range(num_satellites):

            part_data = particle_data.load_particle_data(sim_info, sample_satellites_i[j])
            bound_particles_only = part_data.select_bound_particles(sim_info, index_satellites_i[j], part_data.stars.ids)
            if len(bound_particles_only) < 5: continue # We cannot do much with only few particles

            elements_list = ("Fe_H", "C_Fe", "N_Fe", "Mg_Fe", "O_Fe", "Si_Fe", "Ne_Fe")

            data = part_data.stars.calculate_abundances(elements_list)
            Fe_H = np.append(Fe_H, data["Fe_H"][bound_particles_only])
            C_Fe = np.append(C_Fe, data["C_Fe"][bound_particles_only])
            N_Fe = np.append(N_Fe, data["N_Fe"][bound_particles_only])
            Mg_Fe = np.append(Mg_Fe, data["Mg_Fe"][bound_particles_only])
            O_Fe = np.append(O_Fe, data["O_Fe"][bound_particles_only])
            Si_Fe = np.append(Si_Fe, data["Si_Fe"][bound_particles_only])
            Ne_Fe = np.append(Ne_Fe, data["Ne_Fe"][bound_particles_only])

            num_parts = len(bound_particles_only)
            host_halo_indx = np.append(host_halo_indx, np.ones(num_parts) * halo_indx)
            host_subhalo_indx = np.append(host_subhalo_indx, np.ones(num_parts) * index_satellites_i[j])
            host_galaxy_stellar_mass = np.append(host_galaxy_stellar_mass, np.ones(num_parts) * Ms[j])

    return {"Fe_H":Fe_H, "C_Fe":C_Fe, "N_Fe":N_Fe, "Mg_Fe":Mg_Fe, "O_Fe":O_Fe,
            "Si_Fe":Si_Fe, "Ne_Fe":Ne_Fe, "host_subhalo_index": host_subhalo_indx,
            "host_halo_index": host_halo_indx, "galaxylogMs": host_galaxy_stellar_mass}


def make_post_processing_satellites(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_galaxy_satellite_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/SatelliteGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HostHaloID', data=data["host_halo_index"])
        f.create_dataset('HostSubHaloID', data=data["host_subhalo_index"])
        f.create_dataset('HostGalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('Fe_H', data=data["Fe_H"])
        f.create_dataset('C_Fe', data=data["C_Fe"])
        f.create_dataset('N_Fe', data=data["N_Fe"])
        f.create_dataset('Mg_Fe', data=data["Mg_Fe"])
        f.create_dataset('O_Fe', data=data["O_Fe"])
        f.create_dataset('Si_Fe', data=data["Si_Fe"])
        f.create_dataset('Ne_Fe', data=data["Ne_Fe"])
        data_file.close()