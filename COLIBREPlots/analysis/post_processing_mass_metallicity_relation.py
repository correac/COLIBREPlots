import numpy as np
from tqdm import tqdm
import h5py
from simulation import particle_data
from simulation.simulation_data import read_simulation
from simulation.utilities.constants import (Zsolar, O_H_Sun_Asplund,
                                            Fe_H_Sun_Asplund, Mg_Fe_Sun, Fe_H_Sun)

def calculate_abundances_from_particles(sim_info, sample):

    num_sample = len(sample)
    Fe_H = np.ones(num_sample) * (-1)
    O_H_gas_total = np.ones(num_sample) * (-1)
    O_H_gas_diffused = np.ones(num_sample) * (-1)

    Fe_over_H_Sun = 10 ** (Fe_H_Sun_Asplund - 12.)
    num_sample = len(sample)

    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)

        if len(bound_particles_only) > 1:
            mass = part_data.stars.masses.value[bound_particles_only]
            Fe = part_data.stars.iron[bound_particles_only]
            H = part_data.stars.hydrogen[bound_particles_only]

            star_Fe_over_H = Fe / (55.845 * H)
            metallicity_ratio = np.sum(star_Fe_over_H * mass) / np.sum(mass)
            Fe_H[i] = metallicity_ratio / Fe_over_H_Sun

        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.gas.ids)
        if len(bound_particles_only) > 1:
            mass = part_data.gas.masses.value[bound_particles_only]
            O = part_data.gas.oxygen[bound_particles_only]
            H = part_data.gas.hydrogen[bound_particles_only]
            T = part_data.gas.temperature[bound_particles_only]
            nH = part_data.gas.hydrogen_number_density[bound_particles_only]
            select_cold_dense = np.where((T<10**4.5) & (nH>0.1))[0]
            gas_O_over_H_total = O / (16. * H)
            O_H_ratio = np.sum(gas_O_over_H_total[select_cold_dense] * mass[select_cold_dense]) / np.sum(mass[select_cold_dense])
            #O_H_ratio = np.log10(O_H_ratio / Zsolar) + O_H_Sun_Asplund
            O_H_gas_total[i] = O_H_ratio

            O = part_data.gas.oxygen_diffuse[bound_particles_only]
            H = part_data.gas.hydrogen_diffuse[bound_particles_only]
            gas_O_over_H_diffuse = O / (16. * H)
            O_H_ratio = np.sum(gas_O_over_H_diffuse[select_cold_dense] * mass[select_cold_dense]) / np.sum(mass[select_cold_dense])
            #O_H_ratio = np.log10(O_H_ratio / Zsolar) + O_H_Sun_Asplund
            O_H_gas_diffused[i] = O_H_ratio

    return Fe_H, O_H_gas_total, O_H_gas_diffused

def calculate_galaxies_abundances(sim_info):

    #select_sample = np.where(sim_info.halo_data.log10_stellar_mass >= 8)[0]
    #select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]
    #sample = select_sample[select_centrals]
    sample = np.where(sim_info.halo_data.type == 10)[0]

    Z = sim_info.halo_data.metallicity[sample]
    halo_indx_list = sim_info.halo_data.halo_index[sample]
    galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample]
    galaxy_sfr = sim_info.halo_data.sfr[sample]

    if sim_info.soap_name is None:
        Fe_H, O_H_gas_total, O_H_gas_diffused = calculate_abundances_from_particles(sim_info, sample)
    else:
        Fe_H = sim_info.halo_data.Fe_over_H[sample]
        O_H_gas_total = sim_info.halo_data.log10_O_H_gas_total_plus_twelve[sample]
        O_H_gas_diffused = sim_info.halo_data.log10_O_H_gas_diffuse_plus_twelve[sample]


    return {"Fe_H":Fe_H, "O_H_gas_total":O_H_gas_total, "O_H_gas_diffused":O_H_gas_diffused,
            "halo_index":halo_indx_list, "Z":Z, "galaxylogMs": galaxy_stellar_mass, "galaxySFR": galaxy_sfr}


def make_post_processing_galaxies_metallicities(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_galaxies_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/Galaxies_mass_metallicity_relation_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HaloID', data=data["halo_index"])
        f.create_dataset('GalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('GalaxySFR', data=data["galaxySFR"])
        f.create_dataset('Metallicity', data=data["Z"])
        f.create_dataset('O_H_gas_total', data=data["O_H_gas_total"])
        f.create_dataset('O_H_gas_diffused', data=data["O_H_gas_diffused"])
        f.create_dataset('Fe_H', data=data["Fe_H"])
        data_file.close()

def calculate_alpha_enhancement_from_particles(sim_info, sample):

    num_sample = len(sample)
    Fe_H = np.ones(num_sample) * (-1)
    Mg_Fe = np.ones(num_sample) * (-1)

    num_sample = len(sample)

    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)

        if len(bound_particles_only) > 1:
            mass = part_data.stars.masses.value[bound_particles_only]
            Mg = part_data.stars.magnesium[bound_particles_only]
            Fe = part_data.stars.iron[bound_particles_only]
            H = part_data.stars.hydrogen[bound_particles_only]

            star_Fe = np.sum(Fe * mass)
            star_H = np.sum(H * mass)
            metallicity_ratio = star_Fe / star_H
            Fe_H[i] = np.log10(metallicity_ratio) - Fe_H_Sun

            star_Mg = np.sum(Mg * mass)
            metallicity_ratio = star_Mg / star_Fe
            Mg_Fe[i] = np.log10(metallicity_ratio) - Mg_Fe_Sun

    return Fe_H, Mg_Fe


def calculate_alpha_enhancement_galaxies(sim_info):

    sample = np.where(sim_info.halo_data.type == 10)[0]

    Z = sim_info.halo_data.metallicity[sample]
    halo_indx_list = sim_info.halo_data.halo_index[sample]
    galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample]
    galaxy_sfr = sim_info.halo_data.sfr[sample]

    if sim_info.soap_name is None:
        _, Mg_Fe = calculate_alpha_enhancement_from_particles(sim_info, sample)
    else:
        Mg_Fe = sim_info.halo_data.log_Mg_over_Fe[sample]

    return {"Mg_Fe":Mg_Fe, "halo_index":halo_indx_list,
            "Z":Z, "galaxylogMs": galaxy_stellar_mass, "galaxySFR": galaxy_sfr}

def make_post_processing_galaxies_MgFe(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_alpha_enhancement_galaxies(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/Galaxies_MgFe_mass_relation_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HaloID', data=data["halo_index"])
        f.create_dataset('GalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('GalaxySFR', data=data["galaxySFR"])
        f.create_dataset('Metallicity', data=data["Z"])
        f.create_dataset('Mg_Fe', data=data["Mg_Fe"])
        data_file.close()
