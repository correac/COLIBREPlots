import numpy as np
import swiftsimio as sw
import unyt
import h5py
from typing import Tuple
from astropy.cosmology import Planck13 as cosmo
from .utilities import constants
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

class load_star_particle_data:
    """
    Class containing particles properties
    """

    def __init__(
            self, sim_info, CoP, CoV, size
    ):
        """
        Parameters
        ----------
        """
        mask = sw.mask(f"{sim_info.directory}/{sim_info.snapshot_name}")

        # region is a 3x2 list [[left, right], [bottom, top], [front, back]]
        region = [[-0.5 * b + o, 0.5 * b + o] for b, o in zip(size, CoP)]

        # Constrain the mask
        mask.constrain_spatial(region)

        # Now load the snapshot with this mask
        data = sw.load(f"{sim_info.directory}/{sim_info.snapshot_name}", mask=mask)

        self.ids = data.stars.particle_ids.value
        self.n_parts = len(self.ids)

        self.coordinates = data.stars.coordinates.to("Mpc") * sim_info.a
        self.coordinates -= CoP * sim_info.a  # centering

        self.velocities = data.stars.velocities.to("km/s") - CoV
        self.masses = data.stars.masses.to("Msun")

        self.metal_mass_fractions = data.stars.metal_mass_fractions.value

        # self.baryon_max_soft = 0.5 * sim_info.baryon_max_soft * np.ones(self.n_parts)
        # self.weighted_mass = self.masses * (1.2348 / self.baryon_max_soft) ** 3
        #
        # self.birth_scale_factors = data.stars.birth_scale_factors.value

        # Reading abundances
        self.hydrogen = data.stars.element_mass_fractions.hydrogen.value
        self.helium = data.stars.element_mass_fractions.helium.value
        self.carbon = data.stars.element_mass_fractions.carbon.value
        self.nitrogen = data.stars.element_mass_fractions.nitrogen.value
        self.neon = data.stars.element_mass_fractions.neon.value
        self.magnesium = data.stars.element_mass_fractions.magnesium.value
        self.oxygen = data.stars.element_mass_fractions.oxygen.value
        self.iron = data.stars.element_mass_fractions.iron.value
        self.silicon = data.stars.element_mass_fractions.silicon.value
        self.europium = data.stars.element_mass_fractions.europium.value
        self.barium = data.stars.element_mass_fractions.barium.value
        self.strontium = data.stars.element_mass_fractions.strontium.value


    def calculate_abundances(self, args):

        low_limit = 1e-8

        if len(args) > 0:
            data = {}
            for arg in args:
                if arg == "O_H":
                    O_H = np.clip(self.oxygen, low_limit, None) / self.hydrogen
                    O_H = np.log10(O_H) - constants.O_H_Sun
                    data[arg] = O_H
                elif arg == "Fe_H":
                    Fe_H = np.clip(self.iron, low_limit, None) / self.hydrogen
                    Fe_H = np.log10(Fe_H) - constants.Fe_H_Sun
                    data[arg] = Fe_H
                elif arg == "Mg_H":
                    Mg_H = np.clip(self.magnesium, low_limit, None) / self.hydrogen
                    Mg_H = np.log10(Mg_H) - constants.Mg_H_Sun
                    data[arg] = Mg_H
                elif arg == "C_Fe":
                    C_Fe = np.clip(self.carbon, low_limit, None) / np.clip(self.iron, low_limit, None)
                    C_Fe = np.log10(C_Fe) - constants.C_Fe_Sun
                    data[arg] = C_Fe
                elif arg == "N_Fe":
                    N_Fe = np.clip(self.nitrogen, low_limit, None) / np.clip(self.iron, low_limit, None)
                    N_Fe = np.log10(N_Fe) - constants.N_Fe_Sun
                    data[arg] = N_Fe
                elif arg == "N_O":
                    N_O = np.clip(self.nitrogen, low_limit, None) / np.clip(self.oxygen, low_limit, None)
                    N_O = np.log10(N_O) - constants.N_O_Sun
                    data[arg] = N_O
                elif arg == "C_O":
                    C_O = np.clip(self.carbon, low_limit, None) / np.clip(self.oxygen, low_limit, None)
                    C_O = np.log10(C_O) - constants.C_O_Sun
                    data[arg] = C_O
                elif arg == "O_Fe":
                    O_Fe = np.clip(self.oxygen, low_limit, None) / np.clip(self.iron, low_limit, None)
                    O_Fe = np.log10(O_Fe) - constants.O_Fe_Sun
                    data[arg] = O_Fe
                elif arg == "Mg_Fe":
                    Mg_Fe = np.clip(self.magnesium, low_limit, None) / np.clip(self.iron, low_limit, None)
                    Mg_Fe = np.log10(Mg_Fe) - constants.Mg_Fe_Sun
                    data[arg] = Mg_Fe
                elif arg == "O_Mg":
                    O_Mg = np.log10(self.oxygen / self.magnesium) - constants.O_Mg_Sun
                    O_Mg[self.magnesium == 0] = -10  # set lower limit
                    O_Mg[self.oxygen == 0] = -10  # set lower limit
                    data[arg] = O_Mg
                elif arg == "CN_Fe":
                    CN_Mg = np.log10((self.carbon + self.nitrogen) / self.magnesium) - constants.CN_Mg_Sun
                    CN_Mg[self.magnesium == 0] = -10  # set lower limit
                    data[arg] = CN_Mg
                elif arg == "Si_Fe":
                    Si_Fe = np.clip(self.silicon, low_limit, None) / np.clip(self.iron, low_limit, None)
                    Si_Fe = np.log10(Si_Fe) - constants.Si_Fe_Sun
                    data[arg] = Si_Fe
                elif arg == "Ne_Fe":
                    Ne_Fe = np.clip(self.neon, low_limit, None) / np.clip(self.iron, low_limit, None)
                    Ne_Fe = np.log10(Ne_Fe) - constants.Ne_Fe_Sun
                    data[arg] = Ne_Fe
                elif arg == "Eu_Fe":
                    Eu_Fe = np.log10(self.europium / self.iron) - constants.Eu_Fe_Sun
                    Eu_Fe[self.iron == 0] = -10
                    Eu_Fe[self.europium == 0] = -10
                    data[arg] =  Eu_Fe
                elif arg == "Ba_Fe":
                    Ba_Fe = np.log10(self.barium / self.iron) - constants.Ba_Fe_Sun
                    Ba_Fe[self.barium == 0] = -10
                    Ba_Fe[self.iron == 0] = -10
                    data[arg] = Ba_Fe
                elif arg == "Sr_Fe":
                    Sr_Fe = np.log10(self.strontium / self.iron) - constants.Sr_Fe_Sun
                    Sr_Fe[self.strontium == 0] = -10
                    Sr_Fe[self.iron == 0] = -10
                    data[arg] = Sr_Fe
                else:
                    raise AttributeError(f"Unknown variable: {arg}!")

        return data

    def calculate_stellar_ages(self, part_list):
        # When needed we can calculate the particles age

        stars_birthz = (
                1.0 / self.birth_scale_factors[part_list] - 1.0
        )

        cosmic_age_z0 = cosmo.age(0.0).value
        cosmic_age = cosmo.age(stars_birthz).value
        age = np.ones(self.n_parts) * cosmic_age_z0 - cosmic_age
        return age

    def calculate_elements_masses(self, sim_info, index):

        # The full metadata object is available from within the mask
        size = unyt.unyt_array([1, 1, 1], 'Mpc')

        x = sim_info.halo_data.xminpot[index]
        y = sim_info.halo_data.yminpot[index]
        z = sim_info.halo_data.zminpot[index]

        CoP = unyt.unyt_array([x, y, z], "Mpc") / sim_info.a  # to comoving

        mask = sw.mask(f"{sim_info.directory}/{sim_info.snapshot_name}")

        # region is a 3x2 list [[left, right], [bottom, top], [front, back]]
        region = [[-0.5 * b + o, 0.5 * b + o] for b, o in zip(size, CoP)]

        # Constrain the mask
        mask.constrain_spatial(region)

        # Now load the snapshot with this mask
        data = sw.load(f"{sim_info.directory}/{sim_info.snapshot_name}", mask=mask)

        self.hydrogen_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 0].value * data.stars.masses.to("Msun").value
        self.helium_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 1].value * data.stars.masses.to("Msun").value
        self.carbon_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 2].value * data.stars.masses.to("Msun").value
        self.nitrogen_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 3].value * data.stars.masses.to("Msun").value
        self.oxygen_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 4].value * data.stars.masses.to("Msun").value
        self.neon_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 5].value * data.stars.masses.to("Msun").value
        self.magnesium_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 6].value * data.stars.masses.to("Msun").value
        self.silicon_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 7].value * data.stars.masses.to("Msun").value
        self.iron_mass_from_agb = data.stars.element_mass_fractions_from_agb[:, 8].value * data.stars.masses.to("Msun").value

        self.hydrogen_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 0].value * data.stars.masses.to("Msun").value
        self.helium_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 1].value * data.stars.masses.to("Msun").value
        self.carbon_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 2].value * data.stars.masses.to("Msun").value
        self.nitrogen_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 3].value * data.stars.masses.to("Msun").value
        self.oxygen_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 4].value * data.stars.masses.to("Msun").value
        self.neon_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 5].value * data.stars.masses.to("Msun").value
        self.magnesium_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 6].value * data.stars.masses.to("Msun").value
        self.silicon_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 7].value * data.stars.masses.to("Msun").value
        self.iron_mass_from_snii = data.stars.element_mass_fractions_from_snii[:, 8].value * data.stars.masses.to("Msun").value

        self.hydrogen_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 0].value * data.stars.masses.to("Msun").value
        self.helium_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 1].value * data.stars.masses.to("Msun").value
        self.carbon_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 2].value * data.stars.masses.to("Msun").value
        self.nitrogen_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 3].value * data.stars.masses.to("Msun").value
        self.oxygen_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 4].value * data.stars.masses.to("Msun").value
        self.neon_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 5].value * data.stars.masses.to("Msun").value
        self.magnesium_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 6].value * data.stars.masses.to("Msun").value
        self.silicon_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 7].value * data.stars.masses.to("Msun").value
        self.iron_mass_from_snia = data.stars.element_mass_fractions_from_snia[:, 8].value * data.stars.masses.to("Msun").value

        self.europium_mass_from_nsm = data.stars.mass_fractions_from_nsm.value * data.stars.masses.to("Msun").value
        self.europium_mass_from_cejsn = data.stars.mass_fractions_from_cejsn.value * data.stars.masses.to("Msun").value
        self.europium_mass_from_collapsar = data.stars.mass_fractions_from_collapsar.value * data.stars.masses.to("Msun").value


class load_gas_particle_data:
    """
    Class containing particles properties
    """

    def __init__(
            self, sim_info, CoP, CoV, size
    ):
        """
        Parameters
        ----------
        """
        mask = sw.mask(f"{sim_info.directory}/{sim_info.snapshot_name}")

        # region is a 3x2 list [[left, right], [bottom, top], [front, back]]
        region = [[-0.5 * b + o, 0.5 * b + o] for b, o in zip(size, CoP)]

        # Constrain the mask
        mask.constrain_spatial(region)

        # Now load the snapshot with this mask
        data = sw.load(f"{sim_info.directory}/{sim_info.snapshot_name}", mask=mask)

        self.ids = data.gas.particle_ids.value
        self.n_parts = len(self.ids)

        self.coordinates = data.gas.coordinates.to("Mpc") * sim_info.a
        self.coordinates -= CoP * sim_info.a  # centering

        self.velocities = data.gas.velocities.to("km/s") - CoV
        self.masses = data.gas.masses.to("Msun")

        self.smoothing_length = data.gas.smoothing_lengths.value * sim_info.a * sim_info.to_kpc_units

        XH = data.gas.element_mass_fractions.hydrogen.value
        gas_HI = data.gas.species_fractions.HI.value
        gas_H2 = data.gas.species_fractions.H2.value * 2.0

        self.HI_mass = gas_HI * XH * self.masses
        self.H2_mass = gas_H2 * XH * self.masses

        self.star_formation_rates = (data.gas.star_formation_rates.value
                                     * sim_info.to_Msun_units / sim_info.to_yr_units)

        self.hydrogen_number_density = data.gas.densities.to("g/cm**3") / constants.mH_in_cgs * XH

        self.temperature = data.gas.temperatures.to("K").value

        self.metal_mass_fractions = data.gas.metal_mass_fractions.value
        self.oxygen = data.gas.element_mass_fractions.oxygen.value     # Total (Dust + Diffuse)
        self.hydrogen = data.gas.element_mass_fractions.hydrogen.value # Total (Dust + Diffuse)
        self.carbon = data.gas.element_mass_fractions.carbon.value     # Total (Dust + Diffuse)
        self.nitrogen = data.gas.element_mass_fractions.nitrogen.value # Total (Dust + Diffuse)
        self.iron = data.gas.element_mass_fractions.iron.value         # Total (Dust + Diffuse)

        self.oxygen_diffuse = data.gas.element_mass_fractions_diffuse[:,4].value     # Diffuse
        self.hydrogen_diffuse = data.gas.element_mass_fractions_diffuse[:,0].value   # Diffuse
        self.carbon_diffuse = data.gas.element_mass_fractions_diffuse[:,2].value     # Diffuse
        self.nitrogen_diffuse = data.gas.element_mass_fractions_diffuse[:,3].value   # Diffuse
        self.iron_diffuse = data.gas.element_mass_fractions_diffuse[:,8].value       # Diffuse


class load_particle_data:
    """
    Class containing particles properties
    """

    def __init__(
            self, sim_info, index,
    ):
        """
        Parameters
        ----------
        """

        # The full metadata object is available from within the mask
        size = unyt.unyt_array([1, 1, 1], 'Mpc')

        x = sim_info.halo_data.xminpot[index]
        y = sim_info.halo_data.yminpot[index]
        z = sim_info.halo_data.zminpot[index]

        vx = sim_info.halo_data.vxminpot[index]
        vy = sim_info.halo_data.vyminpot[index]
        vz = sim_info.halo_data.vzminpot[index]

        CoP = unyt.unyt_array([x, y, z], "Mpc") / sim_info.a  # to comoving

        CoV = unyt.unyt_array([vx, vy, vz], "km/s")

        self.stars = load_star_particle_data(sim_info, CoP, CoV, size)

        self.gas = load_gas_particle_data(sim_info, CoP, CoV, size)


    def select_bound_particles(self, sim_info, halo_index, ids):
        """
        Select particles that are gravitationally bound to halo
        Parameters
        ----------
        halo_id: int
        Halo id from the catalogue
        Returns
        -------
        Output: Tuple[np.ndarray, np.ndarray]
        A tuple containing ids of the stellar particles and gas particles
        """
        particles_file = h5py.File(f"{sim_info.directory}/{sim_info.catalogue_particles}", "r")
        group_file = h5py.File(f"{sim_info.directory}/{sim_info.catalogue_groups}", "r")

        halo_start_position = group_file["Offset"][halo_index.astype('int')]
        try:
            halo_end_position = group_file["Offset"][halo_index.astype('int') + 1]
        except IndexError:
            return np.array([-1])

        particle_ids_in_halo = particles_file["Particle_IDs"][halo_start_position:halo_end_position]

        _, _, mask = np.intersect1d(
            particle_ids_in_halo,
            ids,
            assume_unique=True,
            return_indices=True,
        )

        # Ensure that there are no negative indices
        mask = mask[mask > 0]

        return mask

    def calculate_galactocentric_radius(self, momentum, pos):

        face_on_rotation_matrix = rotation_matrix_from_vector(momentum)

        x, y, z = np.matmul(face_on_rotation_matrix, pos.T)
        radius = np.sqrt(x**2 + y**2)

        return radius, z

