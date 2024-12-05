import numpy as np
import unyt
from velociraptor import load
from .utilities.constants import Fe_H_Sun_Asplund

class HaloCatalogue:
    """
    General class containing halo properties
    """

    def __init__(
        self, path_to_catalogue: str, 
        galaxy_min_stellar_mass: unyt.array.unyt_quantity,
        galaxy_min_gas_mass: unyt.array.unyt_quantity,
    ):
        """
        Parameters
        ----------
        path_to_catalogue: str
        Path to the catalogue with halo properties

        galaxy_min_stellar_mass: unyt.array.unyt_quantity
        Minimum stellar mass in units of Msun. Objects whose stellar mass is lower than this
        threshold are disregarded. Same for gas mass.
        """

        self.path_to_catalogue = path_to_catalogue

        # Load catalogue using velociraptor python library
        catalogue = load(self.path_to_catalogue)

        # Selecting central galaxies whose stellar mass is larger than
        # 'galaxy_min_stellar_mass'
        # mask = np.logical_and(
        #     catalogue.apertures.mass_star_50_kpc >= galaxy_min_stellar_mass,
        #     catalogue.apertures.mass_gas_50_kpc > galaxy_min_gas_mass
        # )
        mask = np.where(
            (catalogue.apertures.mass_star_50_kpc >= galaxy_min_stellar_mass) &
            (catalogue.apertures.mass_gas_50_kpc > galaxy_min_gas_mass)
        )[0]

        # Compute the number of haloes following the selection mask
        self.number_of_haloes = len(mask)

        # Ids of haloes satisfying the selection criterion
        self.halo_index = mask.copy()

        # Log10 stellar mass in units of Msun
        self.log10_stellar_mass = np.log10(
            catalogue.apertures.mass_star_50_kpc.to("Msun").value[mask]
        )
        # Log10 gas mass in units of Msun
        self.log10_gas_mass = np.log10(
            catalogue.apertures.mass_gas_50_kpc.to("Msun").value[mask]
        )
        # Log10 halo mass in units of Msun
        self.log10_halo_mass = np.log10(
            catalogue.masses.mass_200crit.to("Msun").value[mask]
        )

        self.concentration = catalogue.concentration.cnfw.value[mask]

        self.virial_radius = catalogue.radii.r_200crit.to("kpc").value[mask]

        self.scale_radius = self.virial_radius / self.concentration

        # Galaxy type, either central (=10) or satellite (>10)
        self.type = catalogue.structure_type.structuretype.value[mask]

        # Half mass radius in units of kpc (stars)
        self.half_mass_radius_star = catalogue.radii.r_halfmass_star.to("kpc").value[
            mask
        ]
        # Half mass radius in units of kpc (gas)
        self.half_mass_radius_gas = catalogue.radii.r_halfmass_gas.to("kpc").value[mask]

        # Star formation rate in units of Msun/yr
        self.sfr = (
            catalogue.apertures.sfr_gas_30_kpc.value[mask] * 10227144.8879616 / 1e9
        )

        # Metallicity of star-forming gas
        self.metallicity_gas_sfr = catalogue.apertures.zmet_gas_sf_50_kpc.value[mask]

        # Metallicity of all gas
        self.metallicity_gas = catalogue.apertures.zmet_gas_50_kpc.value[mask]

        self.metallicity = catalogue.apertures.zmet_star_50_kpc.value[mask]

        # Ids of haloes satisfying the selection criterion
        self.halo_ids = np.array([i for i in range(len(mask)) if mask[i] == True])

        self.xminpot = catalogue.positions.xcminpot.to("Mpc").value[mask]
        self.yminpot = catalogue.positions.ycminpot.to("Mpc").value[mask]
        self.zminpot = catalogue.positions.zcminpot.to("Mpc").value[mask]

        self.vxminpot = catalogue.velocities.vxcminpot.to("km/s").value[mask]
        self.vyminpot = catalogue.velocities.vycminpot.to("km/s").value[mask]
        self.vzminpot = catalogue.velocities.vzcminpot.to("km/s").value[mask]

        self.satellite_flag = catalogue.satellites[mask]
        self.central_flag = catalogue.centrals[mask]

        self.vmax = catalogue.velocities.vmax.to("km/s").value[mask]


def calculate_morphology(pos, vel, mass):

    # Compute distances
    distancesDATA = np.sqrt(np.sum(pos ** 2, axis=1))

    # Restrict particles -
    extract = distancesDATA < 30.0
    pos = pos[extract, :]
    vel = vel[extract, :]
    mass = mass[extract]

    distancesDATA = distancesDATA[extract]

    Mstar = np.sum(mass) # compute total in-aperture stellar mass

    # Compute 30kpc CoM to Sub CoM velocty offset & recenter
    dvVmass = np.sum(mass[:, np.newaxis] * vel, axis=0) / Mstar
    vel -= dvVmass

    # Compute momentum
    smomentums = np.cross(pos, vel)
    momentum = np.sum(mass[:, np.newaxis] * smomentums, axis=0)
    normed_momentum = momentum / np.linalg.norm(momentum)

    # Compute rotational velocities
    smomentumz = np.sum(momentum * smomentums / np.linalg.norm(momentum), axis=1)
    cyldistances = (distancesDATA ** 2 - np.sum(momentum * pos / np.linalg.norm(momentum), axis=1) ** 2)
    cyldistances = np.sqrt(np.abs(cyldistances))

    if len(cyldistances[cyldistances > 0]) > 0:
        cylmin = np.min(cyldistances[cyldistances > 0])
        cyldistances[cyldistances == 0] = cylmin
        vrots = smomentumz / cyldistances
    else:
        vrots = smomentumz

    # Compute kappa_co
    Mvrot2 = np.sum((mass * vrots ** 2)[vrots > 0])
    kappa_co = Mvrot2 / np.sum(mass * (np.linalg.norm(vel, axis=1)) ** 2)

    return kappa_co, normed_momentum


class SOAP:
    """
    General class containing halo properties
    """

    def __init__(
        self, path_to_catalogue: str,
        galaxy_min_stellar_mass: unyt.array.unyt_quantity,
        galaxy_min_gas_mass: unyt.array.unyt_quantity,
    ):
        """
        Parameters
        ----------
        path_to_catalogue: str
        Path to the catalogue with halo properties

        galaxy_min_stellar_mass: unyt.array.unyt_quantity
        Minimum stellar mass in units of Msun. Objects whose stellar mass is lower than this
        threshold are disregarded. Same for gas mass.
        """

        self.path_to_soap_catalogue = path_to_catalogue

        # Load catalogue using velociraptor python library
        catalogue = load(self.path_to_soap_catalogue)

        mass_star_50_kpc = catalogue.get_quantity("apertures.mass_star_50_kpc")

        mass_gas_50_kpc = catalogue.get_quantity("apertures.mass_gas_50_kpc")

        mask = np.where(
            (mass_star_50_kpc >= galaxy_min_stellar_mass) &
            (mass_gas_50_kpc > galaxy_min_gas_mass)
        )[0]

        # Compute the number of haloes following the selection mask
        self.number_of_haloes = len(mask)

        # Ids of haloes satisfying the selection criterion
        self.halo_index = mask.copy()

        # Log10 stellar mass in units of Msun
        self.log10_stellar_mass = np.log10(
            mass_star_50_kpc.to("Msun").value[mask]
        )
        # Log10 gas mass in units of Msun
        self.log10_gas_mass = np.log10(
            mass_gas_50_kpc.to("Msun").value[mask]
        )
        del mass_gas_50_kpc

        # Log10 halo mass in units of Msun
        self.log10_halo_mass = np.log10(
            catalogue.get_quantity("masses.mass_200crit").to("Msun").value[mask]
        )

        self.virial_radius = catalogue.get_quantity("radii.r_200crit").to("kpc").value[mask]

        # Galaxy type, either central (=10) or satellite (>10)
        self.type = catalogue.get_quantity("structure_type.structuretype").value[mask]

        # Half mass radius in units of kpc (stars)
        self.half_mass_radius_star = catalogue.get_quantity("radii.r_halfmass_star").to("kpc").value[mask]

        # Half mass radius in units of kpc (gas)
        self.half_mass_radius_gas = catalogue.get_quantity("radii.r_halfmass_gas").to("kpc").value[mask]

        # Star formation rate in units of Msun/yr
        self.sfr = catalogue.get_quantity("apertures.sfr_gas_50_kpc").to("Msun/yr").value[mask]

        # Metallicity of star-forming gas
        self.metallicity_gas_sfr = catalogue.get_quantity("apertures.zmet_gas_sf_50_kpc").value[mask]

        # Metallicity of all gas
        self.metallicity_gas = catalogue.get_quantity("apertures.zmet_gas_50_kpc").value[mask]

        self.metallicity = catalogue.get_quantity("apertures.zmet_star_50_kpc").value[mask]

        O_over_H_total_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_O_over_H_total_times_gas_mass_50_kpc"
        ).to("Msun").value[mask]

        O_over_H_diffuse_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_O_over_H_times_gas_mass_50_kpc"
            ).to("Msun").value[mask]

        colddense_mass = catalogue.get_quantity(
            f"cold_dense_gas_properties.cold_dense_gas_mass_50_kpc"
        ).to("Msun").value[mask]

        self.log10_O_H_gas_total_plus_twelve = np.log10(
            O_over_H_total_times_gas_mass / colddense_mass) + 12
        self.log10_O_H_gas_total_plus_twelve = np.where(
            np.isnan(self.log10_O_H_gas_total_plus_twelve), -1, self.log10_O_H_gas_total_plus_twelve)

        self.log10_O_H_gas_diffuse_plus_twelve = np.log10(
            O_over_H_diffuse_times_gas_mass / colddense_mass) + 12
        self.log10_O_H_gas_diffuse_plus_twelve = np.where(
            np.isnan(self.log10_O_H_gas_diffuse_plus_twelve), -1, self.log10_O_H_gas_diffuse_plus_twelve)

        del O_over_H_total_times_gas_mass,  O_over_H_diffuse_times_gas_mass

        N_over_O_total_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_N_over_O_total_times_gas_mass_50_kpc"
        ).to("Msun").value[mask]

        N_over_O_diffuse_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_N_over_O_times_gas_mass_50_kpc"
            ).to("Msun").value[mask]

        self.log10_N_O_gas_total = np.log10(
            N_over_O_total_times_gas_mass / colddense_mass)
        self.log10_N_O_gas_total = np.where(
            np.isnan(self.log10_N_O_gas_total), -1, self.log10_N_O_gas_total)

        self.log10_N_O_gas_diffuse = np.log10(
            N_over_O_diffuse_times_gas_mass / colddense_mass)
        self.log10_N_O_gas_diffuse = np.where(
            np.isnan(self.log10_N_O_gas_diffuse), -1, self.log10_N_O_gas_diffuse)

        del N_over_O_total_times_gas_mass,  N_over_O_diffuse_times_gas_mass

        C_over_O_total_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_C_over_O_total_times_gas_mass_50_kpc"
        ).to("Msun").value[mask]

        C_over_O_diffuse_times_gas_mass = catalogue.get_quantity(
            "lin_element_ratios_times_masses.lin_C_over_O_times_gas_mass_50_kpc"
            ).to("Msun").value[mask]

        self.log10_C_O_gas_total = np.log10(
            C_over_O_total_times_gas_mass / colddense_mass)
        self.log10_C_O_gas_total = np.where(
            np.isnan(self.log10_C_O_gas_total), -1, self.log10_C_O_gas_total)

        self.log10_C_O_gas_diffuse = np.log10(
            C_over_O_diffuse_times_gas_mass / colddense_mass)
        self.log10_C_O_gas_diffuse = np.where(
            np.isnan(self.log10_C_O_gas_diffuse), -1, self.log10_C_O_gas_diffuse)

        del colddense_mass, C_over_O_total_times_gas_mass,  C_over_O_diffuse_times_gas_mass

        lin_Fe_over_H_times_star_mass = catalogue.get_quantity(
            f"lin_element_ratios_times_masses.lin_Fe_over_H_times_star_mass_50_kpc"
        ).to("Msun").value[mask]

        Fe_over_H_Sun = 10 ** (Fe_H_Sun_Asplund - 12.)
        self.Fe_over_H = lin_Fe_over_H_times_star_mass / mass_star_50_kpc.to("Msun").value[mask] / Fe_over_H_Sun

        # Magnesium & Iron mass in stars
        Mass_Mg = catalogue.get_quantity(f"element_masses_in_stars.magnesium_mass_50_kpc").value[mask]
        Mass_Fe = catalogue.get_quantity(f"element_masses_in_stars.iron_mass_50_kpc").value[mask]

        # Avoid zeroes
        mask_Mg = np.logical_and(Mass_Fe > 0.0, Mass_Mg > 0.0)

        # Floor value for the field below
        floor_value = -5

        X_Mg_to_X_Fe_solar = 0.55
        Mg_over_Fe = floor_value * np.ones_like(Mass_Fe)
        Mg_over_Fe[mask_Mg] = np.log10(Mass_Mg[mask_Mg] / Mass_Fe[mask_Mg]) - np.log10(X_Mg_to_X_Fe_solar)
        self.log_Mg_over_Fe = Mg_over_Fe

        # Ids of haloes satisfying the selection criterion
        self.halo_ids = np.array([i for i in range(len(mask)) if mask[i] == True])

        self.xminpot = catalogue.get_quantity(
            "positions.xcminpot").to("Mpc").value[mask]

        self.yminpot = catalogue.get_quantity(
            "positions.ycminpot").to("Mpc").value[mask]

        self.zminpot = catalogue.get_quantity(
            "positions.zcminpot").to("Mpc").value[mask]

        self.vxminpot = catalogue.get_quantity(
            "velocities.vxc").to("km/s").value[mask]

        self.vyminpot = catalogue.get_quantity(
            "velocities.vyc").to("km/s").value[mask]

        self.vzminpot = catalogue.get_quantity(
            "velocities.vzc").to("km/s").value[mask]

        self.vmax = catalogue.get_quantity(
            "velocities.vmax").to("km/s").value[mask]