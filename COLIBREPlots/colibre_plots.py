"""
Welcome to COLIBRE PLOTS python package, just a plotting pipeline for the COLIBRE simulations
"""

from argumentparser import ArgumentParser
from plotter.abundance_ratios import plot_abundance_ratios_comparison, plot_abundance_ratios_single_run
from plotter.scatter_analysis_satellites import plot_scatter_of_MW_satellites
# from plotter.scatter_analysis_MW import plot_scatter_MW_with_radius, plot_stellar_distribution
# from plotter.scatter_analysis_MW import plot_APOGEE, plot_abundance_ratio_MgFe
from analysis.post_processing_MW_abundances import make_post_processing_MW
#from analysis.post_processing_MW_abundances import make_post_processing_MW_gas_abundances
from analysis.post_processing_satellite_abundances import make_post_processing_satellites
#from analysis.post_processing_pie_charts import make_post_processing_MW_for_abundance_contribution
from plotter.pie_charts import plot_pie_charts
#from plotter.gas_abundance import plot_gas_abundance_gradient
from analysis.post_processing_mass_metallicity_relation import make_post_processing_galaxies_metallicities
from plotter.mass_metallicity_relation import plot_mass_metallicity_relation

if __name__ == "__main__":

    config_parameters = ArgumentParser()
    # make_post_processing_MW_gas_abundances(config_parameters)
    # plot_gas_abundance_gradient(config_parameters)

    # plot_abundance_ratio_MgFe(config_parameters)
    # make_post_processing_satellites(config_parameters)

    # plot_abundance_ratios_comparison(config_parameters)
    # plot_abundance_ratio_MgFe(config_parameters)
    # plot_scatter_of_MW_satellites(config_parameters)
    # plot_scatter_MW_with_radius(config_parameters)
    # plot_stellar_distribution(config_parameters)
    # plot_APOGEE()

    # Let's make Fig. 2 of draft
    # make_post_processing_MW(config_parameters)
    # make_post_processing_MW_for_abundance_contribution(config_parameters)
    plot_pie_charts(config_parameters)
    # plot_abundance_ratios_single_run(config_parameters)

    # make_post_processing_galaxies_metallicities(config_parameters)
    # plot_mass_metallicity_relation(config_parameters)
