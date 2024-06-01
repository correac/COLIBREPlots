import numpy as np
import unyt
from velociraptor import load


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