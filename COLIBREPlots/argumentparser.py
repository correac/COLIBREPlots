import argparse
import unyt
from typing import List, Optional


class ArgumentParser(object):
    """
    Class for handeling arguments that are passed to the script
    """

    # List of snapshots that are to be processed.
    snapshot_list: List[str]
    # List of catalogues (in the same order as snapshots) to be processed.
    catalogue_list: List[str]
    # List of directories that contain the above snapshots and catalogues
    directory_list: List[str]
    # List of representative names for the snapshots; may be a list of Nones
    name_list: List[Optional[str]]
    # Directory to output the figure to
    output_directory: str
    # Number of inputs to the script.
    number_of_inputs: int
    # Number of galaxies to make individual plots for
    number_of_galaxies: int
    # Minimum stellar mass. Haloes with smaller stellar masses won't be processed
    min_stellar_mass: unyt.array.unyt_quantity

    def __init__(self):

        parser = argparse.ArgumentParser(
            description="""General argument parser for morphology pipeline."""
        )

        parser.add_argument(
            "-s",
            "--snapshots",
            help="Snapshot list. Do not include directory. Example: snapshot_0000.hdf5",
            type=str,
            required=True,
            nargs="*",
        )

        parser.add_argument(
            "-c",
            "--catalogues",
            help=(
                "Catalogue list. Do not include directory. Example: "
                "catalogue_0000.properties"
            ),
            type=str,
            required=True,
            nargs="*",
        )

        parser.add_argument(
            "-d",
            "--input-directories",
            help="Input directory list. Catalogue and snapshot are in this directory.",
            type=str,
            required=True,
            nargs="*",
        )

        parser.add_argument(
            "-n",
            "--run-names",
            help="Names of the runs for placement in legends.",
            type=str,
            required=False,
            nargs="*",
        )

        parser.add_argument(
            "-m",
            "--min-stellar-mass",
            help="Minimum stellar mass in units of Msun. Haloes whose stellar mass is smaller than "
            "this threshold won't be processed.",
            type=float,
            required=False,
            default=1e6,
        )

        parser.add_argument(
            "-o",
            "--output-directory",
            help="Output directory for the produced figure.",
            type=str,
            required=True,
        )

        args = parser.parse_args()

        self.snapshot_list = args.snapshots
        self.catalogue_list = args.catalogues
        self.directory_list = args.input_directories

        if args.run_names is not None:
            self.name_list = args.run_names
        else:
            self.name_list = [None] * len(self.directory_list)

        self.output_directory = args.output_directory

        self.number_of_inputs = len(args.snapshots)
        self.min_stellar_mass = unyt.unyt_quantity(args.min_stellar_mass, "Msun")

        print("")
        print("Welcome to COLIBRE Chemistry Plots!")
        print("Here are your parse arguments:")
        print("---------------------\n")
        print(f"Snapshot list: {self.snapshot_list}")
        print(f"Catalogue list: {self.catalogue_list}")
        print(f"Input directory list: {self.directory_list}")
        print(f"Run names: {self.name_list}")
        print(f"Output directory: {self.output_directory}")
        print(f"Number of runs: {self.number_of_inputs}")
        print("")
