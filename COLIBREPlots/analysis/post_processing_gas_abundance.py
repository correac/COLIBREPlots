import numpy as np
import h5py
from tqdm import tqdm
from simulation.simulation_data import read_simulation
from simulation import particle_data
from simulation.utilities.constants import mH_in_cgs, mC_in_cgs, mN_in_cgs, mO_in_cgs

def calculate_cno_abundances(sim_info):

    select_sample = np.where(sim_info.halo_data.log10_stellar_mass >= 10.5)[0]
    select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]
    sample = select_sample[select_centrals]

    sample_halo_indx_list = sim_info.halo_data.halo_index[sample]
    sample_galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample]

    num_sample = len(sample)

    O_H_gas_total = []
    O_H_gas_diffused = []
    N_O_gas_total = []
    N_O_gas_diffused = []
    C_O_gas_total = []
    C_O_gas_diffused = []
    halo_indx_list = []
    galaxy_stellar_mass = []

    for i in tqdm(range(num_sample)):

        halo_indx = sample_halo_indx_list[i]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.gas.ids)

        if len(bound_particles_only) > 10:
            O = part_data.gas.oxygen[bound_particles_only]
            H = part_data.gas.hydrogen[bound_particles_only]
            C = part_data.gas.carbon[bound_particles_only]
            N = part_data.gas.nitrogen[bound_particles_only]

            sfr = part_data.gas.star_formation_rates[bound_particles_only]
            T = part_data.gas.temperature[bound_particles_only]
            nH = part_data.gas.hydrogen_number_density[bound_particles_only]
            select_cold_dense_starforming = np.where( (T<10**4.5) & (nH>0.1) & (sfr>10) )[0]

            if len(select_cold_dense_starforming) > 10:

                O = O[select_cold_dense_starforming]
                H = H[select_cold_dense_starforming]
                C = C[select_cold_dense_starforming]
                N = N[select_cold_dense_starforming]

                gas_O_over_H = (O / mO_in_cgs) / (H / mH_in_cgs)
                gas_O_over_H = np.log10(gas_O_over_H) + 12
                gas_N_O = (N / mN_in_cgs) / (O / mO_in_cgs)
                gas_N_O = np.log10(gas_N_O)
                gas_C_O = (C / mC_in_cgs) / (O / mO_in_cgs)
                gas_C_O = np.log10(gas_C_O)

                O_H_gas_total = np.append(O_H_gas_total, gas_O_over_H)
                N_O_gas_total = np.append(N_O_gas_total, gas_N_O)
                C_O_gas_total = np.append(C_O_gas_total, gas_C_O)

                index = np.ones(len(select_cold_dense_starforming))
                halo_indx_list = np.append(halo_indx_list, index * halo_indx)
                galaxy_stellar_mass = np.append(galaxy_stellar_mass, index * sample_galaxy_stellar_mass[i])

                O = part_data.gas.oxygen_diffuse[bound_particles_only]
                H = part_data.gas.hydrogen_diffuse[bound_particles_only]
                C = part_data.gas.carbon_diffuse[bound_particles_only]
                N = part_data.gas.nitrogen_diffuse[bound_particles_only]
                O = O[select_cold_dense_starforming]
                H = H[select_cold_dense_starforming]
                C = C[select_cold_dense_starforming]
                N = N[select_cold_dense_starforming]

                gas_O_over_H = (O / mO_in_cgs) / (H / mH_in_cgs)
                gas_O_over_H = np.log10(gas_O_over_H) + 12
                gas_N_O = (N / mN_in_cgs) / (O / mO_in_cgs)
                gas_N_O = np.log10(gas_N_O)
                gas_C_O = (C / mC_in_cgs) / (O / mO_in_cgs)
                gas_C_O = np.log10(gas_C_O)

                O_H_gas_diffused = np.append(O_H_gas_diffused, gas_O_over_H)
                N_O_gas_diffused = np.append(N_O_gas_diffused, gas_N_O)
                C_O_gas_diffused = np.append(C_O_gas_diffused, gas_C_O)


    return {"O_H_gas_total":O_H_gas_total, "O_H_gas_diffused":O_H_gas_diffused,
            "N_O_gas_total": N_O_gas_total, "N_O_gas_diffused": N_O_gas_diffused,
            "C_O_gas_total": C_O_gas_total, "C_O_gas_diffused": C_O_gas_diffused,
            "halo_index":halo_indx_list, "galaxylogMs": galaxy_stellar_mass}

def make_post_processing_cno(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_cno_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/CNO_gas_abundance_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HaloID', data=data["halo_index"])
        f.create_dataset('GalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('O_H_gas_total', data=data["O_H_gas_total"])
        f.create_dataset('O_H_gas_diffused', data=data["O_H_gas_diffused"])
        f.create_dataset('N_O_gas_total', data=data["N_O_gas_total"])
        f.create_dataset('N_O_gas_diffused', data=data["N_O_gas_diffused"])
        f.create_dataset('C_O_gas_total', data=data["C_O_gas_total"])
        f.create_dataset('C_O_gas_diffused', data=data["C_O_gas_diffused"])
        data_file.close()