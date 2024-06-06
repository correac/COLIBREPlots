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

    C = []
    N = []
    Mg = []
    O = []
    Si = []
    Ne = []
    Fe = []

    C_agb = []
    N_agb = []
    Mg_agb = []
    O_agb = []
    Si_agb = []
    Ne_agb = []
    Fe_agb = []

    C_snii = []
    N_snii = []
    Mg_snii = []
    O_snii = []
    Si_snii = []
    Ne_snii = []
    Fe_snii = []

    C_snia = []
    N_snia = []
    Mg_snia = []
    O_snia = []
    Si_snia = []
    Ne_snia = []
    Fe_snia = []

    Eu_nsm = []
    Eu_coll = []
    Eu_cejsn = []

    galactocentric_radius = []
    galactocentric_zcoordinate = []

    host_halo_indx = []
    host_galaxy_kappa_parameter = []
    host_galaxy_stellar_mass = []


    for i in tqdm(range(num_sample)):

        halo_indx = sim_info.halo_data.halo_index[sample[i]]
        part_data = particle_data.load_particle_data(sim_info, sample[i])
        bound_particles_only = part_data.select_bound_particles(sim_info, halo_indx, part_data.stars.ids)
        galaxy_stellar_mass = sim_info.halo_data.log10_stellar_mass[sample[i]]

        mass = part_data.stars.masses.value[bound_particles_only]
        pos = part_data.stars.coordinates.value[bound_particles_only, :] * 1e3 # kpc
        vel = part_data.stars.velocities.value[bound_particles_only]

        kappa, momentum = calculate_morphology(pos, vel, mass)
        gradius, zcoord = part_data.calculate_galactocentric_radius(momentum, pos)

        part_data.stars.calculate_elements_masses(sim_info, sample[i])

        C = np.append(C, part_data.stars.carbon[bound_particles_only] * mass)
        N = np.append(N, part_data.stars.nitrogen[bound_particles_only] * mass)
        Mg = np.append(Mg, part_data.stars.magnesium[bound_particles_only] * mass)
        O = np.append(O, part_data.stars.oxygen[bound_particles_only] * mass)
        Si = np.append(Si, part_data.stars.silicon[bound_particles_only] * mass)
        Ne = np.append(Ne, part_data.stars.neon[bound_particles_only] * mass)
        Fe = np.append(Fe, part_data.stars.iron[bound_particles_only] * mass)

        C_agb = np.append(C_agb, part_data.stars.carbon_mass_from_agb[bound_particles_only])
        N_agb = np.append(N_agb, part_data.stars.nitrogen_mass_from_agb[bound_particles_only])
        Mg_agb = np.append(Mg_agb, part_data.stars.magnesium_mass_from_agb[bound_particles_only])
        O_agb = np.append(O_agb, part_data.stars.oxygen_mass_from_agb[bound_particles_only])
        Si_agb = np.append(Si_agb, part_data.stars.silicon_mass_from_agb[bound_particles_only])
        Ne_agb = np.append(Ne_agb, part_data.stars.neon_mass_from_agb[bound_particles_only])
        Fe_agb = np.append(Fe_agb, part_data.stars.iron_mass_from_agb[bound_particles_only])

        C_snii = np.append(C_snii, part_data.stars.carbon_mass_from_snii[bound_particles_only])
        N_snii = np.append(N_snii, part_data.stars.nitrogen_mass_from_snii[bound_particles_only])
        Mg_snii = np.append(Mg_snii, part_data.stars.magnesium_mass_from_snii[bound_particles_only])
        O_snii = np.append(O_snii, part_data.stars.oxygen_mass_from_snii[bound_particles_only])
        Si_snii = np.append(Si_snii, part_data.stars.silicon_mass_from_snii[bound_particles_only])
        Ne_snii = np.append(Ne_snii, part_data.stars.neon_mass_from_snii[bound_particles_only])
        Fe_snii = np.append(Fe_snii, part_data.stars.iron_mass_from_snii[bound_particles_only])

        C_snia = np.append(C_snia, part_data.stars.carbon_mass_from_snia[bound_particles_only])
        N_snia = np.append(N_snia, part_data.stars.nitrogen_mass_from_snia[bound_particles_only])
        Mg_snia = np.append(Mg_snia, part_data.stars.magnesium_mass_from_snia[bound_particles_only])
        O_snia = np.append(O_snia, part_data.stars.oxygen_mass_from_snia[bound_particles_only])
        Si_snia = np.append(Si_snia, part_data.stars.silicon_mass_from_snia[bound_particles_only])
        Ne_snia = np.append(Ne_snia, part_data.stars.neon_mass_from_snia[bound_particles_only])
        Fe_snia = np.append(Fe_snia, part_data.stars.iron_mass_from_snia[bound_particles_only])

        Eu_nsm = np.append(Eu_nsm, part_data.stars.europium_mass_from_nsm[bound_particles_only])
        Eu_coll = np.append(Eu_coll, part_data.stars.europium_mass_from_collapsar[bound_particles_only])
        Eu_cejsn = np.append(Eu_cejsn, part_data.stars.europium_mass_from_cejsn[bound_particles_only])

        galactocentric_radius = np.append(galactocentric_radius, gradius)
        galactocentric_zcoordinate = np.append(galactocentric_zcoordinate, zcoord)

        num_parts = len(bound_particles_only)
        host_halo_indx = np.append(host_halo_indx, np.ones(num_parts) * halo_indx)
        host_galaxy_kappa_parameter = np.append(host_galaxy_kappa_parameter, np.ones(num_parts) * kappa)
        host_galaxy_stellar_mass = np.append(host_galaxy_stellar_mass, np.ones(num_parts) * galaxy_stellar_mass)

    return {"C":C, "N":N, "Mg":Mg, "O":O, "Si":Si, "Ne":Ne, "Fe":Fe,
            "C_agb":C_agb, "N_agb":N_agb, "Mg_agb":Mg_agb, "O_agb":O_agb,
            "Si_agb":Si_agb, "Ne_agb":Ne_agb, "Fe_agb":Fe_agb,
            "C_snii": C_snii, "N_snii": N_snii, "Mg_snii": Mg_snii, "O_snii": O_snii,
            "Si_snii": Si_snii, "Ne_snii": Ne_snii, "Fe_snii": Fe_snii,
            "C_snia": C_snia, "N_snia": N_snia, "Mg_snia": Mg_snia, "O_snia": O_snia,
            "Si_snia": Si_snia, "Ne_snia": Ne_snia, "Fe_snia": Fe_snia,
            "Eu_nsm": Eu_nsm, "Eu_coll": Eu_coll, "Eu_cejsn":Eu_cejsn,
            "Rgal": galactocentric_radius, "zgal":galactocentric_zcoordinate,
            "halo_index": host_halo_indx, "galaxyKappa": host_galaxy_kappa_parameter,
            "galaxylogMs": host_galaxy_stellar_mass}



def make_post_processing_MW_for_abundance_contribution(config_parameters):

    for i in range(config_parameters.number_of_inputs):

        sim_info = read_simulation(config_parameters, i)
        data = calculate_MW_abundances(sim_info)

        # Output data
        output_file = f"{sim_info.output_path}/MWGalaxies_contribution_abundances_" + sim_info.simulation_name + ".hdf5"
        data_file = h5py.File(output_file, 'w')
        f = data_file.create_group('Data')
        f.create_dataset('HostHaloID', data=data["halo_index"])
        f.create_dataset('HostGalaxyKappa', data=data["galaxyKappa"])
        f.create_dataset('HostGalaxylogMstellar', data=data["galaxylogMs"])

        f.create_dataset('C', data=data["C"])
        f.create_dataset('N', data=data["N"])
        f.create_dataset('Mg', data=data["Mg"])
        f.create_dataset('O', data=data["O"])
        f.create_dataset('Si', data=data["Si"])
        f.create_dataset('Ne', data=data["Ne"])
        f.create_dataset('Fe', data=data["Fe"])

        f.create_dataset('C_AGB', data=data["C_agb"])
        f.create_dataset('N_AGB', data=data["N_agb"])
        f.create_dataset('Mg_AGB', data=data["Mg_agb"])
        f.create_dataset('O_AGB', data=data["O_agb"])
        f.create_dataset('Si_AGB', data=data["Si_agb"])
        f.create_dataset('Ne_AGB', data=data["Ne_agb"])
        f.create_dataset('Fe_AGB', data=data["Fe_agb"])

        f.create_dataset('C_SNII', data=data["C_snii"])
        f.create_dataset('N_SNII', data=data["N_snii"])
        f.create_dataset('Mg_SNII', data=data["Mg_snii"])
        f.create_dataset('O_SNII', data=data["O_snii"])
        f.create_dataset('Si_SNII', data=data["Si_snii"])
        f.create_dataset('Ne_SNII', data=data["Ne_snii"])
        f.create_dataset('Fe_SNII', data=data["Fe_snii"])

        f.create_dataset('C_SNIa', data=data["C_snia"])
        f.create_dataset('N_SNIa', data=data["N_snia"])
        f.create_dataset('Mg_SNIa', data=data["Mg_snia"])
        f.create_dataset('O_SNIa', data=data["O_snia"])
        f.create_dataset('Si_SNIa', data=data["Si_snia"])
        f.create_dataset('Ne_SNIa', data=data["Ne_snia"])
        f.create_dataset('Fe_SNIa', data=data["Fe_snia"])

        f.create_dataset('Eu_NSM', data=data["Eu_nsm"])
        f.create_dataset('Eu_Collapsar', data=data["Eu_coll"])
        f.create_dataset('Eu_CEJSN', data=data["Eu_cejsn"])

        f.create_dataset('GalactocentricRadius', data=data["Rgal"])
        f.create_dataset('GalactocentricZ', data=data["zgal"])
        data_file.close()