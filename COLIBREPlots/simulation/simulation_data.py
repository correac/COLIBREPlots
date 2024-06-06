from typing import List, Union, Tuple, Dict
import glob
import os
from .utilities import constants
from .halo_catalogue import HaloCatalogue, SOAP
import swiftsimio
from argumentparser import ArgumentParser


def read_simulation(config: ArgumentParser, num_arg: int):

    # Fetch relevant input parameters from list
    directory = config.directory_list[num_arg]
    snapshot = config.snapshot_list[num_arg]
    catalogue = config.catalogue_list[num_arg]
    soap = config.soap_list[num_arg]
    sim_name = config.name_list[num_arg]
    output = config.output_directory

    # Load all data and save it in SimInfo class
    sim_info = SimInfo(
        directory=directory,
        snapshot=snapshot,
        catalogue=catalogue,
        soap=soap,
        output=output,
        name=sim_name
    )

    return sim_info

class SimInfo:

    def __init__(
        self,
        directory: str,
        snapshot: str,
        catalogue: str,
        soap: str,
        output: str,
        name: Union[str, None]
    ):
        """
        Parameters
        ----------

        directory: str
        Run directory

        snapshot: str
        Name of the snapshot file

        catalogue: str
        Name of the catalogue file

        name:
        Name of the run

        galaxy_min_stellar_mass: array
        """

        self.directory = directory
        self.output_path = output

        if snapshot is not None:
            self.snapshot_name = snapshot
            base_name = "".join([s for s in self.snapshot_name if not s.isdigit() and s != "_"])
            base_name = os.path.splitext(base_name)[0]
            self.snapshot_base_name = base_name

            # Load snapshot via swiftsimio
            self.snapshot = swiftsimio.load(f"{self.directory}/{self.snapshot_name}")


            # Conversion from internal units to kpc
            self.to_kpc_units = (
                self.snapshot.metadata.internal_code_units["Unit length in cgs (U_L)"][0]
                / constants.kpc
            )

            # Conversion from internal units to Msun
            self.to_Msun_units = (
                self.snapshot.metadata.internal_code_units["Unit mass in cgs (U_M)"][0]
                / constants.Msun
            )

            # Conversion from internal units to Myr
            self.to_Myr_units = (
                self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
                / constants.Myr
            )

            # Conversion from internal units to yr
            self.to_yr_units = (
                self.snapshot.metadata.internal_code_units["Unit time in cgs (U_t)"][0]
                / constants.yr
            )

            self.Zsolar = constants.Zsolar

            # Box size of the simulation in kpc
            self.boxSize = self.snapshot.metadata.boxsize.to("kpc").value[0]

            # Cosmic scale factor
            self.a = self.snapshot.metadata.scale_factor

            self.hubble_time_Gyr = self.snapshot.metadata.cosmology.hubble_time.value

            self.Omega_m = self.snapshot.metadata.cosmology.Om0

            # No curvature
            self.Omega_l = self.Omega_m

            # Maximum softening for baryons
            self.baryon_max_soft = (
                self.snapshot.metadata.gravity_scheme[
                    "Maximal physical baryon softening length  [internal units]"
                ][0]
                * self.to_kpc_units
            )

            # Smallest galaxies we consider here are those with at least 100 star/gas particles
            self.min_stellar_mass = 10 * self.snapshot.stars.masses[0].to('Msun')
            self.min_gas_mass = 10 * self.snapshot.gas.masses[0].to('Msun')

        else:
            print("We don't have a snapshot loaded. Ok?")

        # Fetch the run name if not provided
        if name is not None:
            self.simulation_name = name
        else:
            self.simulation_name = self.snapshot.metadata.run_name

        if catalogue is not None:
            self.catalogue_name = catalogue
            catalogue_base_name = "".join([s for s in self.catalogue_name if not s.isdigit() and s != "_"])
            catalogue_base_name = os.path.splitext(catalogue_base_name)[0]
            self.catalogue_base_name = catalogue_base_name

            # Find the group and particle catalogue files
            self.__find_groups_and_particles_catalogues()

            # Object containing halo properties (from halo catalogue)
            self.halo_data = HaloCatalogue(
                path_to_catalogue=f"{self.directory}/{self.catalogue_name}",
                galaxy_min_stellar_mass=self.min_stellar_mass,
                galaxy_min_gas_mass=self.min_gas_mass,
            )

        elif soap is not None:
            self.soap_name = soap
            self.halo_data = SOAP(
                path_to_catalogue=f"{self.directory}/{self.soap_name}",
                galaxy_min_stellar_mass=self.min_stellar_mass,
                galaxy_min_gas_mass=self.min_gas_mass,
            )

        else:
            print("We don't have a halo catalogue loaded. Ok?")

        print(f"Data from run '{self.simulation_name}' has been loaded! \n")

        return

    def __find_groups_and_particles_catalogues(self) -> None:
        """
        Finds paths to the fields with particles catalogue and groups catalogue
        """

        catalogue_num = "".join([s for s in self.catalogue_name.split('.')[0] if s.isdigit()])
        catalogue_groups_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_groups*"
        )
        catalogue_particles_paths: List[str] = glob.glob(
            f"{self.directory}/*{catalogue_num}.catalog_particles*"
        )

        # We expect one file for particle groups
        if len(catalogue_groups_paths) == 1:
            self.catalogue_groups = catalogue_groups_paths[0].split("/")[-1]
        else:
            raise IOError("Couldn't find catalogue_groups file")

        # We expect two files: one for bound and the other for unbound particles
        if len(catalogue_particles_paths) == 2:
            for path in catalogue_particles_paths:
                if path.find("unbound") == -1:
                    self.catalogue_particles = path.split("/")[-1]
        else:
            raise IOError("Couldn't find catalogue_particles file")

        return

