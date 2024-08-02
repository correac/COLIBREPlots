import numpy as np
from tqdm import tqdm
import h5py
from simulation import particle_data
from simulation.halo_catalogue import calculate_morphology
from simulation.simulation_data import read_simulation

def calculate_MW_abundances(sim_info):

    select_sample = np.where((sim_info.halo_data.log10_halo_mass >= 11.8) &
                             (sim_info.halo_data.log10_halo_mass <= 12.2))[0]

    select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]

    sample = select_sample[select_centrals]
    num_sample = len(sample)

    C_Fe = []
    N_Fe = []
    Mg_Fe = []
    O_Fe = []
    Si_Fe = []
    Ne_Fe = []
    Fe_H = []
    Ba_Fe = []
    Sr_Fe = []
    Eu_Fe = []

    galactocentric_radius = []
    galactocentric_zcoordinate = []

    host_halo_indx = []
    host_galaxy_kappa_parameter = []
    host_galaxy_stellar_mass = []


    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample[i]]

        if sim_info.catalogue_particles is not None:
            bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)

            mass = part_data.stars.masses.value[bound_particles_only]
            pos = part_data.stars.coordinates.value[bound_particles_only, :] * 1e3 # kpc
            vel = part_data.stars.velocities.value[bound_particles_only]

        else:
            # Assume bound particles are those within a 50 kpc aperture
            pos = part_data.stars.coordinates.value[:, :] * 1e3 # kpc
            radius = np.sqrt(np.sum(pos ** 2, axis=1))
            bound_particles_only = np.where(radius <= 50)[0]

            mass = part_data.stars.masses.value[bound_particles_only]
            pos = pos[bound_particles_only, :]
            vel = part_data.stars.velocities.value[bound_particles_only]


        kappa, momentum = calculate_morphology(pos, vel, mass)
        gradius, zcoord = part_data.calculate_galactocentric_radius(momentum, pos)

        elements_list = ("Fe_H", "C_Fe", "N_Fe", "Mg_Fe", "O_Fe", "Si_Fe", "Ne_Fe", "Ba_Fe", "Sr_Fe", "Eu_Fe")

        data = part_data.stars.calculate_abundances(elements_list)
        Fe_H = np.append(Fe_H, data["Fe_H"][bound_particles_only])
        C_Fe = np.append(C_Fe, data["C_Fe"][bound_particles_only])
        N_Fe = np.append(N_Fe, data["N_Fe"][bound_particles_only])
        Mg_Fe = np.append(Mg_Fe, data["Mg_Fe"][bound_particles_only])
        O_Fe = np.append(O_Fe, data["O_Fe"][bound_particles_only])
        Si_Fe = np.append(Si_Fe, data["Si_Fe"][bound_particles_only])
        Ne_Fe = np.append(Ne_Fe, data["Ne_Fe"][bound_particles_only])
        Ba_Fe = np.append(Ba_Fe, data["Ba_Fe"][bound_particles_only])
        Sr_Fe = np.append(Sr_Fe, data["Sr_Fe"][bound_particles_only])
        Eu_Fe = np.append(Eu_Fe, data["Eu_Fe"][bound_particles_only])

        galactocentric_radius = np.append(galactocentric_radius, gradius)
        galactocentric_zcoordinate = np.append(galactocentric_zcoordinate, zcoord)

        num_parts = len(bound_particles_only)
        host_halo_indx = np.append(host_halo_indx, np.ones(num_parts) * halo_indx)
        host_galaxy_kappa_parameter = np.append(host_galaxy_kappa_parameter, np.ones(num_parts) * kappa)
        host_galaxy_stellar_mass = np.append(host_galaxy_stellar_mass, np.ones(num_parts) * galaxy_stellar_mass)

    return {"Fe_H":Fe_H, "C_Fe":C_Fe, "N_Fe":N_Fe, "Mg_Fe":Mg_Fe, "O_Fe":O_Fe,
            "Si_Fe":Si_Fe, "Ne_Fe":Ne_Fe, "Ba_Fe":Ba_Fe, "Sr_Fe":Sr_Fe, "Eu_Fe":Eu_Fe,
            "Rgal": galactocentric_radius, "zgal":galactocentric_zcoordinate,
            "halo_index": host_halo_indx, "galaxyKappa": host_galaxy_kappa_parameter,
            "galaxylogMs": host_galaxy_stellar_mass}


def make_post_processing_MW(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_MW_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/MWGalaxies_stellar_abundances_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HostHaloID', data=data["halo_index"])
        f.create_dataset('HostGalaxyKappa', data=data["galaxyKappa"])
        f.create_dataset('HostGalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('Fe_H', data=data["Fe_H"])
        f.create_dataset('C_Fe', data=data["C_Fe"])
        f.create_dataset('N_Fe', data=data["N_Fe"])
        f.create_dataset('Mg_Fe', data=data["Mg_Fe"])
        f.create_dataset('O_Fe', data=data["O_Fe"])
        f.create_dataset('Si_Fe', data=data["Si_Fe"])
        f.create_dataset('Ne_Fe', data=data["Ne_Fe"])
        f.create_dataset('Ba_Fe', data=data["Ba_Fe"])
        f.create_dataset('Sr_Fe', data=data["Sr_Fe"])
        f.create_dataset('Eu_Fe', data=data["Eu_Fe"])
        f.create_dataset('GalactocentricRadius', data=data["Rgal"])
        f.create_dataset('GalactocentricZ', data=data["zgal"])
        data_file.close()

def calculate_MW_gas_abundances(sim_info):

    select_sample = np.where((sim_info.halo_data.log10_halo_mass >= 11.8) &
                             (sim_info.halo_data.log10_halo_mass <= 12.2))[0]

    select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]

    sample = select_sample[select_centrals]
    num_sample = len(sample)

    C = []
    N = []
    O = []
    Fe = []
    H = []

    C_diffuse = []
    N_diffuse = []
    O_diffuse = []
    Fe_diffuse = []
    H_diffuse = []

    gas_mass = []
    gas_temperature = []
    gas_density = []

    galactocentric_radius = []
    galactocentric_zcoordinate = []

    host_halo_indx = []
    host_galaxy_kappa_parameter = []
    host_galaxy_stellar_mass = []

    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample[i]]

        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)
        mass = part_data.stars.masses.value[bound_particles_only]
        pos = part_data.stars.coordinates.value[bound_particles_only, :] * 1e3 # kpc
        vel = part_data.stars.velocities.value[bound_particles_only]
        kappa, momentum = calculate_morphology(pos, vel, mass)

        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.gas.ids)
        pos = part_data.gas.coordinates.value[bound_particles_only, :] * 1e3 # kpc
        gradius, zcoord = part_data.calculate_galactocentric_radius(momentum, pos)

        mass = part_data.gas.masses.value[bound_particles_only]
        temperature = part_data.gas.temperature[bound_particles_only]
        density = part_data.gas.hydrogen_number_density.value[bound_particles_only]

        Fe = np.append(Fe, part_data.gas.iron[bound_particles_only])
        C = np.append(C, part_data.gas.carbon[bound_particles_only])
        N = np.append(N, part_data.gas.nitrogen[bound_particles_only])
        O = np.append(O, part_data.gas.oxygen[bound_particles_only])
        H = np.append(H, part_data.gas.hydrogen[bound_particles_only])

        Fe_diffuse = np.append(Fe_diffuse, part_data.gas.iron_diffuse[bound_particles_only])
        C_diffuse = np.append(C_diffuse, part_data.gas.carbon_diffuse[bound_particles_only])
        N_diffuse = np.append(N_diffuse, part_data.gas.nitrogen_diffuse[bound_particles_only])
        O_diffuse = np.append(O_diffuse, part_data.gas.oxygen_diffuse[bound_particles_only])
        H_diffuse = np.append(H_diffuse, part_data.gas.hydrogen_diffuse[bound_particles_only])

        gas_mass = np.append(gas_mass, mass)
        gas_temperature = np.append(gas_temperature, temperature)
        gas_density = np.append(gas_density, density)

        galactocentric_radius = np.append(galactocentric_radius, gradius)
        galactocentric_zcoordinate = np.append(galactocentric_zcoordinate, zcoord)

        num_parts = len(bound_particles_only)
        host_halo_indx = np.append(host_halo_indx, np.ones(num_parts) * halo_indx)
        host_galaxy_kappa_parameter = np.append(host_galaxy_kappa_parameter, np.ones(num_parts) * kappa)
        host_galaxy_stellar_mass = np.append(host_galaxy_stellar_mass, np.ones(num_parts) * galaxy_stellar_mass)

    return {"Fe":Fe, "C":C, "N":N, "O":O, "H":H, "Fe_diffuse": Fe_diffuse, "C_diffuse": C_diffuse,
            "N_diffuse": N_diffuse, "O_diffuse": O_diffuse, "H_diffuse": H_diffuse,
            "mass":gas_mass, "temperature":gas_temperature, "density":gas_density,
            "Rgal": galactocentric_radius, "zgal":galactocentric_zcoordinate,
            "halo_index": host_halo_indx, "galaxyKappa": host_galaxy_kappa_parameter,
            "galaxylogMs": host_galaxy_stellar_mass}

def make_post_processing_MW_gas_abundances(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_MW_gas_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/MWGalaxies_gas_abundances_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HostHaloID', data=data["halo_index"])
        f.create_dataset('HostGalaxyKappa', data=data["galaxyKappa"])
        f.create_dataset('HostGalaxylogMstellar', data=data["galaxylogMs"])
        f.create_dataset('Fe', data=data["Fe"])
        f.create_dataset('C', data=data["C"])
        f.create_dataset('N', data=data["N"])
        f.create_dataset('O', data=data["O"])
        f.create_dataset('H', data=data["H"])
        f.create_dataset('Fe_diffuse', data=data["Fe_diffuse"])
        f.create_dataset('C_diffuse', data=data["C_diffuse"])
        f.create_dataset('N_diffuse', data=data["N_diffuse"])
        f.create_dataset('O_diffuse', data=data["O_diffuse"])
        f.create_dataset('H_diffuse', data=data["H_diffuse"])
        f.create_dataset('Mass', data=data["mass"])
        f.create_dataset('Temperature', data=data["temperature"])
        f.create_dataset('Density', data=data["density"])
        f.create_dataset('GalactocentricRadius', data=data["Rgal"])
        f.create_dataset('GalactocentricZ', data=data["zgal"])
        data_file.close()