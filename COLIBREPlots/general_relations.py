import numpy as np
from simulation.simulation_data import read_simulation
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from argumentparser import ArgumentParser


def read_data(sim_info):

    select_sample = np.where(sim_info.halo_data.log10_stellar_mass >= 7.0)[0]
    select_centrals = np.where(sim_info.halo_data.type[select_sample] == 10)[0]
    sample = select_sample[select_centrals]

    stellar_mass = sim_info.halo_data.log10_stellar_mass[sample]
    halo_mass = sim_info.halo_data.log10_halo_mass[sample]
    sfr = sim_info.halo_data.sfr[sample]
    sizes = sim_info.halo_data.half_mass_radius_star[sample]
    gas_mass = sim_info.halo_data.log10_gas_mass[sample]
    return {"stellar_mass":stellar_mass, "halo_mass":halo_mass,
            "sfr":sfr, "sizes":sizes, "gas_mass":gas_mass}


def plot_median_relation_one_sigma(x, y, color, output_name):

    num_min_per_bin = 5
    bins = np.arange(7, 14, 0.2)
    ind = np.digitize(x, bins)
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ylo, '-', lw=0.3, color=color, zorder=100)
    plt.plot(xm, yhi, '-', lw=0.3, color=color, zorder=100)
    plt.fill_between(xm, ylo, yhi, color=color, alpha=0.3, edgecolor=None, zorder=0)
    plt.plot(xm, ym, '-', lw=3, color='white', zorder=100)

    if output_name is not None:
        plt.plot(xm, ym, '-', lw=2, color=color, label=output_name, zorder=100)
    else:
        plt.plot(xm, ym, '-', lw=2, color=color, zorder=100)


def plot_relations(config_parameters):

    color_list = ['steelblue', 'lightskyblue', 'y', 'salmon']
    names = ['No diffusion', 'Low diffusion', 'Default diffusion', 'High diffusion']

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (6, 4.5),
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.98,
        "lines.markersize": 0.5,
        "lines.linewidth": 0.2,
        "figure.subplot.wspace": 0.33,
        "figure.subplot.hspace": 0.3,
    }
    rcParams.update(params)
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["halo_mass"], data["stellar_mass"], color_list[i], names[i])

    plt.axis([10, 13, 7, 12.5])
    plt.xlabel("$\log_{10}M_{200\mathrm{c}}$ [M$_{\odot}$]")
    plt.ylabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    plt.legend(loc=[0, 0.52], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
               frameon=False, fontsize=13, ncol=1, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 2, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["halo_mass"], data["gas_mass"], color_list[i], sim_info.simulation_name)

    plt.axis([10, 13, 7, 11])
    plt.xlabel("$\log_{10}M_{200\mathrm{c}}$ [M$_{\odot}$]")
    plt.ylabel("$\log_{10}M_{\mathrm{gas}}$ [M$_{\odot}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)

    #######
    ax = plt.subplot(2, 2, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        plot_median_relation_one_sigma(data["stellar_mass"], data["sizes"], color_list[i], sim_info.simulation_name)

    plt.axis([8, 11, 0, 10])
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("$R_{*}$ [kpc]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)


    #######
    ax = plt.subplot(2, 2, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    for i in range(config_parameters.number_of_inputs):
        sim_info = read_simulation(config_parameters, i)
        data = read_data(sim_info)
        select = np.where(data["sfr"]>0)[0]
        plot_median_relation_one_sigma(data["stellar_mass"][select], data["sfr"][select], color_list[i], sim_info.simulation_name)

    plt.axis([8, 11, 0.001, 5])
    plt.yscale('log')
    plt.xlabel("$\log_{10}M_{*}$ [M$_{\odot}$]")
    plt.ylabel("SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # ax.get_xaxis().set_ticklabels([])
    # plt.legend(loc=[0, 0.72], labelspacing=0.05, handlelength=0.5, handletextpad=0.05,
    #            frameon=False, fontsize=10, ncol=2, columnspacing=0.8)



    plt.savefig(config_parameters.output_directory + "general_relations.png", dpi=300)

if __name__ == "__main__":

    config_parameters = ArgumentParser()
    plot_relations(config_parameters)