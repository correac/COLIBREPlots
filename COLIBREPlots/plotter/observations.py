import numpy as np
import h5py
from simulation.utilities import constants
import matplotlib.pylab as plt
import pandas as pd
import scipy.stats as stats
from .util import func_scatter, sample_IMF

def make_hist(x, y, cut, xi, yi):

    selection = np.where(cut)[0]

    # Create a histogram
    h, xedges, yedges = np.histogram2d(
        x[selection], y[selection], bins=(xi, yi), density=True
    )

    return h, xedges, yedges


def make_stellar_abundance_distribution(x, y, R, z, xi, yi):

    # Galactocentric radius (R) in kpc units
    # Galactocentric azimuthal distance (z) in kpc units
    h = np.zeros((len(xi) - 1, len(yi) - 1))

    # We apply masks to select stars in R & z bins
    for Ri in range(0, 9, 3):
        for zi in range(0, 2, 1):
            mask_R = (R >= Ri) & (R < Ri + 3)
            mask_z = (np.abs(z) >= zi) & (np.abs(z) < zi + 1)
            distance_cut = np.logical_and(mask_R, mask_z)

            hist, xedges, yedges = make_hist(x, y, distance_cut, xi, yi)

            # We combine (add) the 6 histograms to give less weight to stars in the solar vicinity.
            # In this manner all stars in the radial/azimuthal bins contribute with equal weight to
            # the final stellar distribution
            h += hist

    return h, xedges, yedges

def read_APOGEE(args):

    input_filename = "./plotter/Observational_data/APOGEE_data.hdf5"
    apogee_dataset = h5py.File(input_filename, "r")
    GalR = apogee_dataset["GalR"][:]
    Galz = apogee_dataset["Galz"][:]
    FE_H = apogee_dataset["FE_H"][:]

    if len(args) > 0:
        data = {}

        for arg in args:
            if arg == "O_H":
                O_FE = apogee_dataset["O_FE"][:]
                O_H = O_FE + FE_H
                data[arg] = O_H + constants.O_over_H_Grevesse07 - constants.O_H_Sun_Asplund

            elif arg == "O_Fe":
                O_FE = apogee_dataset["O_FE"][:]
                O_Fe_Asplund = constants.O_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = O_FE + constants.O_over_Fe_Grevesse07 - O_Fe_Asplund

            elif arg == "Fe_H":
                data[arg] = FE_H + constants.Fe_over_H_Grevesse07 - constants.Fe_H_Sun_Asplund

            elif arg == "Mg_Fe":
                MG_FE = apogee_dataset["MG_FE"][:]
                Mg_Fe_Asplund = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = MG_FE + constants.Mg_over_Fe_Grevesse07 - Mg_Fe_Asplund

            elif arg == "N_Fe":
                N_FE = apogee_dataset["N_FE"][:]
                N_Fe_Asplund = constants.N_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = N_FE + constants.N_over_Fe_Grevesse07 - N_Fe_Asplund

            elif arg == "C_Fe":
                C_FE = apogee_dataset["C_FE"][:]
                C_Fe_Asplund = constants.C_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = C_FE + constants.C_over_Fe_Grevesse07 - C_Fe_Asplund

            elif arg == "Si_Fe":
                SI_FE = apogee_dataset["SI_FE"][:]
                Si_Fe_Asplund = constants.Si_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = SI_FE + constants.Si_over_Fe_Grevesse07 - Si_Fe_Asplund

            else:
                raise AttributeError(f"Unknown variable: {arg}!")

    return GalR, Galz, data[args[0]], data[args[1]]

def plot_contours(GalR, Galz, x, y, cmap):

    xmin = -3
    xmax = 1
    ymin = -1
    ymax = 1

    ngridx = 100
    ngridy = 50

    # Create grid values first.
    xi = np.linspace(xmin, xmax, ngridx)
    yi = np.linspace(ymin, ymax, ngridy)

    # We apply radial & azimuthal cuts, and combine the stellar distributions
    # to give less weight to stars in the solar vicinity. We create a histogram
    # for each distance cut and then combine them
    h, xedges, yedges = make_stellar_abundance_distribution(x, y, GalR, Galz, xi, yi)

    xbins = 0.5 * (xedges[1:] + xedges[:-1])
    ybins = 0.5 * (yedges[1:] + yedges[:-1])

    z = h.T

    binsize = 0.25
    grid_min = np.log10(
        0.5)  # Note that the histograms have been normalized. Therefore this **does not** indicate a minimum of 1 star per bin!
    grid_max = np.log10(np.ceil(h.max()))
    levels = np.arange(grid_min, grid_max, binsize)
    levels = 10 ** levels

    contour = plt.contour(xbins, ybins, z, levels=levels, linewidths=0.5, cmap=cmap, zorder=0)
    contour.collections[0].set_label('APOGEE data')

def plot_APOGEE(args):

    GalR, Galz, x, y = read_APOGEE(args)
    plot_contours(GalR, Galz, x, y, "turbo")
#
# def plot_APOGEE_scatter(args, with_label):
#
#     if args[1] == "O_Fe": delta = np.array([1])
#     if args[1] == "Mg_Fe": delta = np.array([2])
#
#     GalR, Galz, Fe_H, y = read_APOGEE(args)
#
#     low_Z = np.where((Fe_H >= -2) & (Fe_H <= -1))[0]
#     sig_y = np.std(y[low_Z])
#
#     plt.plot([delta-0.3,delta+0.3], [sig_y, sig_y], '-', lw=1, color='lightsteelblue')
#
#     if with_label == True:
#         plt.plot(delta, sig_y, 'X', ms=7, markeredgecolor='black',
#                  markeredgewidth=0.2, color='lightsteelblue', label='APOGEE')
#     else:
#         plt.plot(delta, sig_y, 'X', ms=7, markeredgecolor='black',
#                  markeredgewidth=0.2, color='lightsteelblue')

def read_2process_APOGEE(args, option):

    input_filename = "./plotter/Observational_data/2process_APOGEE_data.hdf5"
    apogee_dataset = h5py.File(input_filename, "r")
    GalR = apogee_dataset["GalR"][:]
    Galz = apogee_dataset["Galz"][:]
    FE_H = apogee_dataset["FE_H_ASPCAP_corrected"][:]

    if len(args) > 0:
        data = {}

        for arg in args:
            if arg == "O_H":
                O_H = apogee_dataset["O_H_" + option + "_corrected"][:]
                data[arg] = O_H + constants.O_over_H_Grevesse07 - constants.O_H_Sun_Asplund

            # elif arg == "O_Fe":
            #     O_FE = apogee_dataset["O_FE"][:]
            #     O_Fe_Asplund = constants.O_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
            #     data[arg] = O_FE + constants.O_over_Fe_Grevesse07 - O_Fe_Asplund
            #
            elif arg == "Fe_H":
                data[arg] = FE_H + constants.Fe_over_H_Grevesse07 - constants.Fe_H_Sun_Asplund
            #
            # elif arg == "Mg_Fe":
            #     MG_FE = apogee_dataset["MG_FE"][:]
            #     Mg_Fe_Asplund = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
            #     data[arg] = MG_FE + constants.Mg_over_Fe_Grevesse07 - Mg_Fe_Asplund
            #
            # elif arg == "N_Fe":
            #     N_FE = apogee_dataset["N_FE"][:]
            #     N_Fe_Asplund = constants.N_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
            #     data[arg] = N_FE + constants.N_over_Fe_Grevesse07 - N_Fe_Asplund
            #
            # elif arg == "C_Fe":
            #     C_FE = apogee_dataset["C_FE"][:]
            #     C_Fe_Asplund = constants.C_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
            #     data[arg] = C_FE + constants.C_over_Fe_Grevesse07 - C_Fe_Asplund

            elif arg == "Si_Fe":
                SI_H = apogee_dataset["SI_H_ASPCAP_corrected"][:]
                Si_Fe_Asplund = constants.Si_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
                data[arg] = SI_H - FE_H + constants.Si_over_Fe_Grevesse07 - Si_Fe_Asplund

            else:
                raise AttributeError(f"Unknown variable: {arg}!")

    return GalR, Galz, data[args[0]], data[args[1]]

def plot_Sit2024(args, option):

    GalR, Galz, x, y = read_2process_APOGEE(args, option)
    plot_contours(GalR, Galz, x, y)

def plot_GALAH(args, with_label=False):

    observational_data = "./plotter/Observational_data/Buder21_data.hdf5"
    GALAH_data = h5py.File(observational_data, "r")
    galah_edges = np.array(GALAH_data["abundance_bin_edges"])
    if args == "C_Fe":
        obs_plane = np.array(GALAH_data["C_enrichment_vs_Fe_abundance"]).T
    elif args == "O_Fe":
        obs_plane = np.array(GALAH_data["O_enrichment_vs_Fe_abundance"]).T
    elif args == "Mg_Fe":
        obs_plane = np.array(GALAH_data["Mg_enrichment_vs_Fe_abundance"]).T
    elif args == "Si_Fe":
        obs_plane = np.array(GALAH_data["Si_enrichment_vs_Fe_abundance"]).T
    elif args == "Ba_Fe":
        obs_plane = np.array(GALAH_data["Ba_enrichment_vs_Fe_abundance"]).T
    elif args == "Eu_Fe":
        obs_plane = np.array(GALAH_data["Eu_enrichment_vs_Fe_abundance"]).T
    obs_plane[obs_plane < 10] = None

    contour = plt.contour(
        np.log10(obs_plane),
        origin="lower",
        extent=[galah_edges[0], galah_edges[-1], galah_edges[0], galah_edges[-1]],
        zorder=100,
        cmap="twilight_shifted",
        linewidths=0.5,
    )
    if with_label == True:
        contour.collections[0].set_label('GALAH data')



def plot_data_Zepeda2022(xval, yval, with_label):
    file = './plotter/Observational_data/Zepeda_2022.txt'
    data = np.loadtxt(file, usecols=(8, 9, 10, 11, 12, 13, 14))
    FeH = data[:, 0]
    if xval == 'Fe_H':
        xdata = FeH
    elif xval == 'Mg_H':
        MgFe = data[:, 1]
        xdata = MgFe + FeH
    else:
        raise AttributeError(f"Unknown x variable: {xval}!")

    if yval == 'C_Fe':
        CFe = data[:, 2]
        ydata = CFe
    elif yval == 'Sr_Fe':
        SrFe = data[:, 3]
        ydata = SrFe
    elif yval == 'Ba_Fe':
        BaFe = data[:, 4]
        ydata = BaFe
    elif yval == 'Eu_Fe':
        EuFe = data[:, 5]
        ydata = EuFe
    elif yval == 'Sr_Mg':
        SrFe = data[:, 3]
        MgFe = data[:, 1]
        ydata = SrFe - MgFe
    elif yval == 'Ba_Mg':
        BaFe = data[:, 4]
        MgFe = data[:, 1]
        ydata = BaFe - MgFe
    elif yval == 'Eu_Mg':
        EuFe = data[:, 5]
        MgFe = data[:, 1]
        ydata = EuFe - MgFe
    else:
        raise AttributeError(f"Unknown y variable: {yval}!")

    if with_label == True:
        plt.plot(xdata, ydata, 'o', ms=2.5,
                 markeredgecolor='black', markerfacecolor='lavender',
                 markeredgewidth=0.2, label='Zepeda et al. (2022)', zorder=0)
    else:
        plt.plot(xdata, ydata, 'o', ms=2.5, markeredgecolor='black', markerfacecolor='lavender',
                 markeredgewidth=0.2, zorder=0)


def plot_data_Gudin2021(xval, yval, with_label):
    file = './plotter/Observational_data/Gudin_2021.txt'
    data = np.loadtxt(file, usecols=(2, 3, 4, 5, 6))
    FeH = data[:, 0]
    xdata = FeH

    if yval == 'C_Fe':
        CFe = data[:, 1]
        ydata = CFe
    elif yval == 'Sr_Fe':
        SrFe = data[:, 2]
        ydata = SrFe
    elif yval == 'Ba_Fe':
        BaFe = data[:, 3]
        ydata = BaFe
    elif yval == 'Eu_Fe':
        EuFe = data[:, 4]
        ydata = EuFe
    else:
        raise AttributeError(f"Unknown y variable: {yval}!")

    if with_label == True:
        plt.plot(xdata, ydata, 'D', ms=2,
                 markeredgecolor='black', markerfacecolor='antiquewhite',
                 markeredgewidth=0.2, label='Gudin et al. (2021)', zorder=0)
    else:
        plt.plot(xdata, ydata, 'D', ms=2,
                 markeredgecolor='black', markerfacecolor='antiquewhite',
                 markeredgewidth=0.2, zorder=0)

def plot_data_Norfolk2019(xval, yval):

    file = './plotter/Observational_data/Norfolk_2019.txt'
    data = np.loadtxt(file)
    FeH = data[:, 0]
    xdata = FeH

    if yval == 'Sr_Fe':
        SrFe = data[:, 4]
        ydata = SrFe
    elif yval == 'Ba_Fe':
        BaFe = data[:, 2]
        ydata = BaFe
    else:
        raise AttributeError(f"Unknown y variable: {yval}!")

    plt.plot(xdata, ydata, '*', color='darkblue', label='Norfolk et al. (2019)')


def load_strontium_data_Zhao():
    file = './plotter/Observational_data/Zhao_2016.txt'
    # Roeder are calculated wrt to Lodder+ solar metallicity
    # Convert to Asplund et al. (2009)
    Fe_H_Sun = 7.50
    Sr_H_Sun = 2.9
    Sr_Fe_Sun = Sr_H_Sun - Fe_H_Sun

    data = np.loadtxt(file)
    FeH= data[:, 0]
    SrFe = data[:, 1] + Sr_Fe_Sun - constants.Sr_Fe_Sun
    return FeH, SrFe

def load_strontium_data_Roeder():
    file = './plotter/Observational_data/Roeder_2014.txt'
    # Roeder are calculated wrt to Asplund+ solar metallicity
    data = np.loadtxt(file)
    FeH = data[:, 0]
    SrFe = data[:, 1]
    return FeH, SrFe

def load_strontium_data_Spite():
    file = './plotter/Observational_data/Spite_2018.txt'
    # Roeder are calculated wrt to Lodder+ solar metallicity
    # We convert Asplund et al. (2009)
    Fe_H_Sun = 7.50
    Sr_H_Sun = 2.9
    Sr_Fe_Sun = Sr_H_Sun - Fe_H_Sun
    data = np.loadtxt(file)
    FeH = data[:, 0]
    SrFe = data[:, 1] + Sr_Fe_Sun  - constants.Sr_Fe_Sun
    return FeH, SrFe

def plot_StrontiumObsData():

    FeH_Ro, SrFe_Ro = load_strontium_data_Roeder()
    FeH_Sp, SrFe_Sp = load_strontium_data_Spite()
    FeH_Zh, SrFe_Zh = load_strontium_data_Zhao()
    plt.plot(FeH_Sp, SrFe_Sp, '>', ms=4, markeredgecolor='black', markerfacecolor='grey',
             markeredgewidth=0.2, label='Spite et al. (2018)', zorder=0)
    plt.plot(FeH_Zh, SrFe_Zh, 'o', ms=3, markeredgecolor='black', markerfacecolor='darkgrey',
             markeredgewidth=0.2, label='Zhao et al. (2016)', zorder=0)
    plt.plot(FeH_Ro, SrFe_Ro, 's', ms=4, markeredgecolor='black', markerfacecolor='grey',
             markeredgewidth=0.2, label='Roederer et al. (2014)', zorder=0)

def plot_observations_Mead_2024():

    data = pd.read_csv("./plotter/Observational_data/Mead_2024.csv").to_numpy()

    sigma_OFe_dSphs = data[1:,1]
    OFe_err_sigma = data[1:,2]
    sigma_MgFe_dSphs = data[1:, 3]
    MgFe_err_sigma = data[1:, 4]
    xp = np.arange(0.9,1.2,0.05)

    marker_list = ['*','v','o','>','X','s','P','D','^']
    name = ['Draco','UMi','Sextans','Fornax','Sculptor','Carina']

    for i in range(6):

        plt.errorbar(xp[i], sigma_OFe_dSphs[i], yerr=OFe_err_sigma[i], marker=marker_list[i],
                     markersize=5, markeredgecolor="black", ls='none', lw=0.5, c='lightsteelblue',
                     markeredgewidth=0.2, label=name[i], zorder=0)

        plt.errorbar(xp[i] * 2, sigma_MgFe_dSphs[i], yerr=MgFe_err_sigma[i] ,marker=marker_list[i],
                     markersize=5, markeredgecolor="black", ls='none', lw=0.5, c='lightsteelblue',
                     markeredgewidth=0.2, zorder=0)


def plot_observations_Kirby_2010():

    marker_list = ['*','v','o','>','X','<','P','D','^']
    name = ['Draco','UMi','Sextans','Fornax','Sculptor','LeoI','LeoII', 'CVnI']

    data = pd.read_csv("./plotter/Observational_data/Kirby_2010.csv").to_numpy()
    dSph = data[:,6]
    FeH = data[:,8]
    name_list = ['Dra','UMi','Sex','For','Scl','LeoI','LeoII','CVnI']
    xp = np.arange(0.92,1.2,0.02)

    for i in range(8):

        select = np.where(dSph == name_list[i])[0]
        FeHi = FeH[select]

        low_metallicity = np.where((FeHi >= -2.5) & (FeHi <= 0.5))[0]
        sigma = np.std(FeHi[low_metallicity])

        plt.errorbar(xp[i], np.array(sigma), marker=marker_list[i], markersize=5,
                     markeredgecolor="black", ls='none', lw=0.5, c='lightsteelblue',
                     markeredgewidth=0.2, label=name[i])


def load_MW_data_with_Mg_Fe():

    file = './plotter/Observational_data/MW.txt'
    data = np.loadtxt(file)
    FeH = data[:, 0]
    MgFe = data[:, 1]

    #compute COLIBRE standard ratios
    Mg_over_Fe = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.67
    Mg_over_H_AG89 = 7.58
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89

    FeH = FeH + Fe_over_H_AG89 - constants.Fe_H_Sun_Asplund
    MgFe = MgFe + Mg_over_Fe_AG89 - Mg_over_Fe

    return FeH, MgFe


def load_MW_data():
    #compute COLIBRE standard ratios
    Fe_over_H = 12. - 4.5
    Mg_over_H = 12. - 4.4
    O_over_H = 12. - 3.31
    Mg_over_Fe = Mg_over_H - Fe_over_H
    O_over_Fe = O_over_H - Fe_over_H

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.67
    Mg_over_H_AG89 = 7.58
    O_over_H_AG89 = 8.93

    # --
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89
    O_over_Fe_AG89 = O_over_H_AG89 - Fe_over_H_AG89

    # MW data
    FeH_MW = []
    OFe_MW = []

    file = './plotter/Observational_data/Koch_2008.txt'
    data = np.loadtxt(file, skiprows=3)
    FeH_koch = data[:, 1] + Fe_over_H_AG89 - Fe_over_H
    OH_koch = data[:, 2]
    OFe_koch = OH_koch - FeH_koch + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_koch)
    OFe_MW = np.append(OFe_MW, OFe_koch)

    file = './plotter/Observational_data/Bai_2004.txt'
    data = np.loadtxt(file, skiprows=3, usecols=[1, 2])
    FeH_bai = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_bai = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_bai)
    OFe_MW = np.append(OFe_MW, OFe_bai)

    file = './plotter/Observational_data/Cayrel_2004.txt'
    data = np.loadtxt(file, skiprows=18, usecols=[2, 6])
    FeH_cayrel = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_cayrel = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_cayrel)
    OFe_MW = np.append(OFe_MW, OFe_cayrel)

    file = './plotter/Observational_data/Israelian_1998.txt'
    data = np.loadtxt(file, skiprows=3, usecols=[1, 3])
    FeH_isra = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_isra = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_isra)
    OFe_MW = np.append(OFe_MW, OFe_isra)

    file = './plotter/Observational_data/Mishenina_1999.txt'
    data = np.loadtxt(file, skiprows=3, usecols=[1, 3])
    FeH_mish = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_mish = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_mish)
    OFe_MW = np.append(OFe_MW, OFe_mish)

    file = './plotter/Observational_data/Zhang_Zhao_2005.txt'
    data = np.loadtxt(file, skiprows=3)
    FeH_zhang = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_zhang = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_zhang)
    OFe_MW = np.append(OFe_MW, OFe_zhang)

    return FeH_MW, OFe_MW

def plot_MW_data(element):

    if element == 'O':
        # Load MW data:
        FeH_MW, OFe_MW = load_MW_data()

        plt.plot(FeH_MW, OFe_MW, 'D', markeredgecolor='black', markerfacecolor='lightgrey',
                 ms=2, markeredgewidth=0.2, label='MW', zorder=0)

    if element == 'Mg':

        # Load MW data:
        FeH_MW, MgFe_MW = load_MW_data_with_Mg_Fe()

        num_min_per_bin = 5
        bins = np.arange(-4, 2, 0.25)
        ind = np.digitize(FeH_MW, bins)
        ym = [np.median(MgFe_MW[ind == i]) for i in range(1, len(bins)) if len(FeH_MW[ind == i]) > num_min_per_bin]
        xm = [np.median(FeH_MW[ind == i]) for i in range(1, len(bins)) if len(FeH_MW[ind == i]) > num_min_per_bin]

        plt.plot(FeH_MW, MgFe_MW, 'D', ms=2, markeredgewidth=0.2,
                 color='lightgrey', markeredgecolor='black', label='MW-data compilation (Venn et al. 2004)', zorder=0)



def load_satellites_data_Mg_Fe():
    # -------------------------------------------------------------------------------
    # alpha-enhancement (Mg/Fe), extracted manually from Tolstoy, Hill & Tosi (2009)
    # -------------------------------------------------------------------------------
    file = './plotter/Observational_data/Fornax.txt'
    data = np.loadtxt(file)
    FeH_fornax = data[:, 0]
    MgFe_fornax = data[:, 1]

    file = './plotter/Observational_data/Sculptor.txt'
    data = np.loadtxt(file)
    FeH_sculptor = data[:, 0]
    MgFe_sculptor = data[:, 1]

    file = './plotter/Observational_data/Sagittarius.txt'
    data = np.loadtxt(file)
    FeH_sagittarius = data[:, 0]
    MgFe_sagittarius = data[:, 1]

    file = './plotter/Observational_data/Carina.txt'
    data = np.loadtxt(file)
    FeH_carina = data[:, 0]
    MgFe_carina = data[:, 1]

    return FeH_fornax, MgFe_fornax, FeH_sculptor, MgFe_sculptor, \
        FeH_sagittarius, MgFe_sagittarius, FeH_carina, MgFe_carina

def load_satellites_data():
    # compute COLIBRE standard ratios
    Fe_over_H = 12. - 4.5
    Mg_over_H = 12. - 4.4
    O_over_H = 12. - 3.31
    Mg_over_Fe = Mg_over_H - Fe_over_H
    O_over_Fe = O_over_H - Fe_over_H

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.67
    Mg_over_H_AG89 = 7.58
    O_over_H_AG89 = 8.93

    # --
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89
    O_over_Fe_AG89 = O_over_H_AG89 - Fe_over_H_AG89

    ## I assume these works use Grevesser & Anders solar metallicity

    file = './plotter/Observational_data/Letarte_2007.txt'
    data = np.loadtxt(file, skiprows=1)
    FeH_fornax = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_fornax = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = './plotter/Observational_data/Sbordone_2007.txt'
    data = np.loadtxt(file, skiprows=1)
    FeH_sg = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_sg = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = './plotter/Observational_data/Koch_2008.txt'
    data = np.loadtxt(file, skiprows=1)
    FeH_ca = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_ca = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = './plotter/Observational_data/Geisler_2005.txt'
    data = np.loadtxt(file, skiprows=3)
    FeH_scu = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_scu = data[:, 4] - data[:, 0] + O_over_Fe_AG89 - O_over_Fe

    return FeH_fornax, OFe_fornax, FeH_sg, OFe_sg, FeH_ca, OFe_ca, FeH_scu, OFe_scu

def plot_satellites(element):

    if element == 'O':
        # Load Satellite data:
        FeH_fornax, OFe_fornax, FeH_sg, OFe_sg, FeH_ca, OFe_ca, FeH_scu, OFe_scu = load_satellites_data()

        plt.plot(FeH_ca, OFe_ca, '>', color='tab:purple', ms=4, label='Carina')
        plt.plot(FeH_scu, OFe_scu, '*', ms=4, color='tab:green', label='Sculptor')
        plt.plot(FeH_fornax, OFe_fornax, 'o', color='tab:orange', ms=4, label='Fornax')
        plt.plot(FeH_sg, OFe_sg, 'v', ms=4, color='crimson', label='Sagittarius')

    if element == 'Mg':
        FeH_fornax, MgFe_fornax, FeH_sculptor, MgFe_sculptor, \
            FeH_sagittarius, MgFe_sagittarius, FeH_carina, MgFe_carina = load_satellites_data_Mg_Fe()
        plt.plot(FeH_carina, MgFe_carina, 'o', color='crimson', ms=3,
                 markeredgecolor='black',  markeredgewidth=0.2,  label='Carina', zorder=0)
        plt.plot(FeH_sculptor, MgFe_sculptor, '>', color='khaki', ms=3,
                 markeredgecolor='black', markeredgewidth=0.2, label='Sculptor', zorder=0)
        plt.plot(FeH_fornax, MgFe_fornax, '<', color='royalblue', ms=3,
                 markeredgecolor='black', markeredgewidth=0.2, label='Fornax', zorder=0)
        plt.plot(FeH_sagittarius, MgFe_sagittarius, '*', ms=3, color='lightblue',
                 markeredgecolor='black', markeredgewidth=0.2, label='Sagittarius', zorder=0)


def plot_Zahid2017():

    Z_solar_obs = 0.0142
    input_filename = "./plotter/Observational_data/Zahid_2017.txt"

    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = raw[:, 1]
    Z_star = 10 ** raw[:, 3] * Z_solar_obs / constants.Zsolar

    mass_bins = np.arange(8.55, 10.95, 0.1)
    mass_bin_centers = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    Z_median, _, _ = stats.binned_statistic(
        M_star, Z_star, statistic="median", bins=mass_bins
    )
    Z_std_up, _, _ = stats.binned_statistic(
        M_star, Z_star, statistic=lambda x: np.percentile(x, 84.0), bins=mass_bins
    )
    Z_std_do, _, _ = stats.binned_statistic(
        M_star, Z_star, statistic=lambda x: np.percentile(x, 16.0), bins=mass_bins
    )

    M_star = 10 ** mass_bin_centers
    Z_star = Z_median
    # Define the scatter as offset from the mean value
    y_scatter = ((Z_median - Z_std_do, Z_std_up - Z_median))
    plt.errorbar(M_star, Z_star, yerr=y_scatter, color='slategray',
                 marker='o', markersize=6, lw=1, markeredgecolor='black',
                 markeredgewidth=0.2, linestyle='none', label='Zahid et al. (2017)')

def plot_gallazi():

    h_obs = 0.7
    h_sim = 0.6777
    Z_solar_obs = 0.02
    input_filename = "./plotter/Observational_data/Gallazzi_2005_ascii.txt"

    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = (10 ** raw[:, 0]  * (h_sim / h_obs) ** -2 * constants.kroupa_to_chabrier_mass)
    Z_median = 10 ** raw[:, 1] * Z_solar_obs / constants.Zsolar
    Z_lo = 10 ** raw[:, 2] * Z_solar_obs / constants.Zsolar
    Z_hi = 10 ** raw[:, 3] * Z_solar_obs / constants.Zsolar

    # Define the scatter as offset from the mean value
    y_scatter = ((Z_median - Z_lo, Z_hi - Z_median))

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='beige',
                 markeredgecolor='black', markeredgewidth=0.2, ecolor='darkgrey', elinewidth=0.7,
                 marker='*', markersize=8, linestyle='none', lw=1, label='Gallazi et al. (2005)')

def plot_Kudritzki():

    Z_solar_obs = 0.0142
    input_filename = "./plotter/Observational_data/Kudritzki_2016.txt"

    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = 10 ** raw[:, 0]
    Z_median = 10 ** raw[:, 1] * Z_solar_obs / constants.Zsolar

    plt.plot(M_star, Z_median, 'x', ms=5, color='grey', label='Kudritzki et al. (2016)', zorder=1000)

def plot_Yates():

    Z_solar_obs = 0.0142
    input_filename = "./plotter/Observational_data/Yates_2021.txt"

    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = 10 ** raw[:, 0]
    Z_median = 10 ** raw[:, 1] * Z_solar_obs / constants.Zsolar
    Z_std_lo = 10 ** raw[:, 2] * Z_solar_obs / constants.Zsolar
    Z_std_hi = 10 ** raw[:, 3] * Z_solar_obs / constants.Zsolar
    y_scatter = ((Z_median - Z_std_lo, Z_std_hi - Z_median))

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='v', markersize=6, linestyle='none', lw=1, label='Yates et al. (2021)')


def plot_Kirby():

    input_filename = "./plotter/Observational_data/Kirby_2013_ascii.dat"

    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = 10 ** raw[:, 0] * constants.kroupa_to_chabrier_mass
    M_star_lo = 10 ** (raw[:, 0] - raw[:, 1]) * constants.kroupa_to_chabrier_mass
    M_star_hi = 10 ** (raw[:, 0] + raw[:, 2]) * constants.kroupa_to_chabrier_mass

    Z_star = 10 ** raw[:, 3]  # Z/Z_sun
    Z_star_lo = 10 ** (raw[:, 3] - raw[:, 4])  # Z/Z_sun
    Z_star_hi = 10 ** (raw[:, 3] + raw[:, 5]) # Z/Z_sun

    # Define the scatter as offset from the mean value
    x_scatter = ((M_star - M_star_lo, M_star_hi - M_star))
    y_scatter = ((Z_star - Z_star_lo, Z_star_hi - Z_star))

    plt.errorbar(M_star, Z_star, xerr=x_scatter, yerr=y_scatter, color='silver',
                 markeredgecolor='black', markeredgewidth=0.2, marker='>',
                 markersize=6, linestyle='none', lw=1, label='Kirby et al. (2013)')

def plot_Panter_2018():

    Mstellar = np.arange(8, 12, 0.2)
    Mstellar = 10**Mstellar

    A = -0.452
    B = 0.572
    Delta = 1.04
    Mc = 9.66
    x = (np.log10(Mstellar) - Mc ) / Delta
    best_fit = 10**(A + B * np.tanh(x))
    Z_solar_obs = 0.02
    best_fit = (best_fit * Z_solar_obs) / constants.Zsolar
    plt.plot(Mstellar, best_fit, '--', lw=2, color='grey', label='Panter et al. (2008)')

def plot_Curti():
    raw = np.loadtxt("./plotter/Observational_data/Curti2020.txt")

    Mstar = 10.0 ** raw[:, 0]
    metal = raw[:, 1]
    metal_error = raw[:, 2]

    plt.errorbar(Mstar, metal, yerr=metal_error, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2, marker='>',
                 markersize=6, linestyle='none', lw=1, label='Curti et al. (2020)')

def plot_Fraser():
    input_filename = "./plotter/Observational_data/Fraser-McKelvie_2021.csv"
    df = pd.read_csv(input_filename, sep=',')
    Mstar = 10**df.values[:,4]
    Zgas = df.values[:,6] # The raw Zgas is 12 + log10 (O/H)

    plt.plot(Mstar, Zgas, 'x', ms=5, mew=0.3, color='grey', label='Fraser-McKelvie et al. (2021)')


def plot_Tremonti():
    h_obs = 0.7
    h_sim = 0.6777

    input_filename = "./plotter/Observational_data/Tremonti_2004_ascii.txt"
    # Read the data
    raw = np.loadtxt(input_filename)
    M_star = 10 ** raw[:, 0] * (h_sim / h_obs) ** -2
    Z_median = raw[:, 3] # 12 + log(O/H)
    Z_lo = raw[:, 2] # 12 + log(O/H)
    Z_hi = raw[:, 4] # 12 + log(O/H)

    # Define the scatter as offset from the mean value
    y_scatter = (Z_median - Z_lo, Z_hi - Z_median)

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='v', markersize=6, linestyle='none', lw=1, label='Tremonti et al. (2004)')


def plot_Andrews():

    # Use the fit (equation 5) with the parameters from table 4
    log10_M_star_min = 7.4
    log10_M_star_max = 10.5
    gamma = 0.640
    log10_M_TO = 8.901
    Z_asm = 8.798

    M_TO = (10 ** log10_M_TO)

    M_star = np.logspace(log10_M_star_min, log10_M_star_max, 25)
    Z_star = (Z_asm - np.log10(1.0 + (M_TO / M_star) ** gamma)) # 12 + log(O/H)

    # Convert the masses to Chabrier IMF
    M_star = M_star * constants.kroupa_to_chabrier_mass

    plt.plot(M_star, Z_star, 'o', ms=5, markeredgecolor='black',
             markeredgewidth=0.2, color='lightsteelblue',label='Andrews $\&$ Martini (2013)')

def plot_SAGA():
    input_filename = "./plotter/Observational_data/SAGA_satellites.txt"
    raw = np.loadtxt(input_filename)

    M_star = 10 ** raw[:, 0]
    Z_median = raw[:, 1]  # 12 + log(O/H)
    Z_lo = raw[:, 2]  # 12 + log(O/H)
    Z_hi = raw[:, 3]  # 12 + log(O/H)

    # Define the scatter as offset from the mean value
    y_scatter = (Z_median - Z_lo, Z_hi - Z_median)

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='grey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='*', markersize=6, linestyle='none', lw=1, label='Geha et al. (2024)')

def plot_Pristine():

    FeH, MgFe = read_Pristine()
    plt.plot(FeH, MgFe, 'x', ms=5, mew=0.3, color='black', label='Pristine Survey')



def plot_Gallazzi_2021():

    raw = np.loadtxt("./plotter/Observational_data/gallazzi_2021_ascii.txt")
    h_sim = 0.6777
    M_star = 10 ** raw[:, 0] * h_sim ** -2

    # Correction factor due to the difference in (X_O/X_Fe)_Sun
    # From Grevesse + (1991) to Asplund+ (2009)

    O_over_H_Gr91 = 8.91

    # We don't take the Fe_over_H_Gr91 value of 7.67 from Grevesse + (1991), which was used
    # in Gallazzi et al. (2021) because we are confident that that value wasn't correct
    # and instead we need to use a more recent value, which is defined in the following.
    # line. We are currently waiting for a confirmation from Gallazzi that we should indeed
    # use 7.48, as opposed to 7.67.
    Fe_over_H_Gr91 = 7.48

    Mg_over_H_Gr91 = 7.58

    Fe_over_H_Asplund09 = 7.50
    Mg_over_H_Asplund09 = 7.6

    Mg_over_Fe_solar_Gr91 = Mg_over_H_Gr91 - Fe_over_H_Gr91
    Mg_over_Fe_solar_Asplund09 = Mg_over_H_Asplund09 - Fe_over_H_Asplund09

    correction = Mg_over_Fe_solar_Gr91 - Mg_over_Fe_solar_Asplund09

    Z_median = (raw[:, 1] + correction)
    Z_lo = (raw[:, 2] + correction)
    Z_hi = (raw[:, 3] + correction)

    # Define the scatter as offset from the mean value
    y_scatter = np.array((Z_median - Z_lo, Z_hi - Z_median))

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='grey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='*', markersize=6, linestyle='none', lw=1, label='Gallazzi et al. (2021)')


def plot_Romero_Gomez_2023():

    # Read the data. Here the linear fit is taken from
    # Romero Gomez et al. (2023); Table C2; first row
    raw = np.arange(8, 12.25, 0.25)
    M_star = 10 ** raw
    Z_star = 0.04 * raw - 0.28
    error_Z_star = 0.01 * raw + 0.09

    # Correction factor due to the difference in (X_O/X_Fe)_Sun
    # From Grevesse & Sauval (1998) to Asplund+ (2009)
    # Note: Romero-Gomez et al. (2023) re-analyze the spectra of
    # each ATLAS-3D galaxy using the MILES/Vazdekis stellar templates library.
    # MILES adopts the stellar libraries of Coelho et al. (2005), who
    # assume the solar abundances of Grevesse & Sauval (1998).
    # For more details see Vazdekis et al. (2015) (https://arxiv.org/pdf/1504.08032.pdf)

    Mg_H_Grevesse98 = 7.58
    Fe_H_Grevesse98 = 7.50
    Mg_Fe_Grevesse98 = Mg_H_Grevesse98 - Fe_H_Grevesse98
    Fe_over_H_Asplund09 = 7.50
    Mg_over_H_Asplund09 = 7.6
    Mg_over_Fe_Asplund09 = Mg_over_H_Asplund09 - Fe_over_H_Asplund09

    element_list = np.array(["MgFe", "OFe"])

    correction = Mg_Fe_Grevesse98 - Mg_over_Fe_Asplund09

    Z_median = (Z_star + correction)
    Z_lo = (Z_star - error_Z_star + correction)
    Z_hi = (Z_star + error_Z_star + correction)

    # Define the scatter as offset from the mean value
    y_scatter = np.array((Z_median - Z_lo, Z_hi - Z_median))

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='o', markersize=4, linestyle='none', lw=1, label='Romero-Gomez et al. (2023) (ATLAS-3D)')

def plot_Romero_Gomez_2023_dwarfs():

    input_filename = "./plotter/Observational_data/RomeroGomez_2023_ascii.txt"
    delimiter = " "

    # Read the data
    raw = np.loadtxt(input_filename, usecols=[1, 2, 3], skiprows=6)
    M_star = 10 ** raw[:, 2]
    select = np.where(raw[:, 2] > 0)[0]  # Some missing data
    M_star = M_star[select]
    alpha_Fe = raw[select, 0]
    error_alpha_Fe = raw[select, 1]

    Z_median = alpha_Fe
    Z_lo = (alpha_Fe - error_alpha_Fe)
    Z_hi = (alpha_Fe + error_alpha_Fe)

    # Define the scatter as offset from the mean value
    y_scatter = np.array((Z_median - Z_lo, Z_hi - Z_median))

    plt.errorbar(M_star, Z_median, yerr=y_scatter, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='v', markersize=4, linestyle='none', lw=1, label='Romero-Gomez et al. (2023)  (LG dwarfs)')

def plot_APOGEE_black(args):

    GalR, Galz, x, y = read_APOGEE(args)
    plot_contours(GalR, Galz, x, y, "bone")

def plot_Hayden():

    # Read the data
    input_filename = "./plotter/Observational_data/HaydenPawson_2022b.txt"

    raw = np.loadtxt(input_filename)
    logOH_median = raw[:, 0] # 12 + log(O/H)
    logNO_median = raw[:, 1] # log(N/O)
    logNO_lo = raw[:, 2] # log(N/O)
    logNO_hi = raw[:, 3] # log(N/O)

    # Define the scatter as offset from the mean value
    y_scatter = np.array((logNO_median - logNO_lo, logNO_hi - logNO_median))
    plt.errorbar(logOH_median, logNO_median, yerr=y_scatter, color='lightblue',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='o', markersize=4, linestyle='none', lw=0.5,
                 label='Hayden-Pawson et al. (2022)')


def plot_Nicolls(option):

    input_filename = "./plotter/Observational_data/Nicholls_2017.txt"

    # Read the data
    raw = np.loadtxt(input_filename)
    logOH = raw[:, 0]  # 12 + log(O/H)
    logCO = raw[:, 1]  # log(C/O)
    logNO = raw[:, 2]  # log(N/O)

    if option == "NO":
        plt.plot(logOH, logNO, '-', color='grey', lw=1,
                 label='Nicolls et al. (2017)')

    if option == "CO":
        plt.plot(logOH, logCO, '-', color='grey', lw=1,
                 label='Nicolls et al. (2017)')



def plot_Berg():
    input_filename = "./plotter/Observational_data/Berg_2020.txt"


    # Read the data
    raw = np.loadtxt(input_filename, usecols=(1, 2, 3, 4), delimiter=",")
    logOH_mean = raw[:, 0] # log(O/H)+12
    logOH_std = raw[:, 1] # log(O/H)+12
    logNO_mean = raw[:, 2] # log(N/O)
    logNO_std = raw[:, 3] # log(N/O)

    # Define the scatter as offset from the mean value
    x_scatter = np.array(logOH_std)
    y_scatter = np.array(logNO_std)

    plt.errorbar(logOH_mean, logNO_mean, yerr=y_scatter, xerr=x_scatter, color='lightgrey',
                 markeredgecolor='black', markeredgewidth=0.2,
                 marker='v', markersize=4, linestyle='none', lw=0.5, label='Berg et al. (2020)')

def plot_mean_yields():

    input_filename = "./outputs/Mean_CCSN_yields.hdf5"
    with h5py.File(input_filename, "r") as file:
        FeH = file["FeH"][:]
        ratios = file["MgFe"][:]

    plt.plot(FeH, ratios, '-.',lw=2, color='turquoise', zorder=1000)

def plot_Donor():

    input_file = "./plotter/Observational_data/Donor2020.txt"

    # Read the data
    raw = np.loadtxt(input_file, usecols=(3, 8))
    MgFe = raw[:, 1]
    FeH = raw[:, 0]

    # Not sure of this actually. I cannot find anywhere what solar abundances they assume
    # Mg_Fe_Asplund = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
    # correction = constants.Mg_over_Fe_Grevesse07 - Mg_Fe_Asplund
    # MgFe += correction


    plt.plot(FeH, MgFe, 'v', ms=3, mew=0.2, color='antiquewhite', markeredgecolor='black', label='MW-Star clusters')


# def plot_Donor_scatter():
#
#     input_file = "./plotter/Observational_data/Donor2020.txt"
#
#     # Read the data
#     raw = np.loadtxt(input_file, usecols=(3, 8))
#     MgFe = raw[:, 1]
#     FeH = raw[:, 0]
#     stars_mass = sample_IMF(len(FeH))
#
#     num_min_per_bin = 2
#     bins = np.arange(-4, 2, 0.25)
#     ind = np.digitize(FeH, bins)
#     ym = [func_scatter(MgFe[ind == i], stars_mass[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
#     xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
#
#     plt.plot(xm, ym, 'v', markeredgecolor='black', markerfacecolor='antiquewhite',
#              ms=4, markeredgewidth=0.2, label='MW-Star clusters', zorder=0)


def plot_Pritzl_2005():

    input_file = "./plotter/Observational_data/Pritzl_2005.txt"

    # Read the data
    raw = np.loadtxt(input_file, usecols=(1, 2))
    MgFe = raw[:, 1]
    FeH = raw[:, 0]

    Mg_Fe_Asplund = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund
    correction = constants.Mg_over_Fe_Grevesse07 - Mg_Fe_Asplund
    MgFe += correction

    plt.plot(FeH, MgFe, 'o', ms=3, mew=0.2, color='mistyrose',
             markeredgecolor='black', label='MW-Star clusters', zorder=0)


# def plot_Pritzl_2005_scatter():
#
#     input_file = "./plotter/Observational_data/Pritzl_2005.txt"
#
#     # Read the data
#     raw = np.loadtxt(input_file, usecols=(1, 2))
#     MgFe = raw[:, 1]
#     FeH = raw[:, 0]
#
#     num_min_per_bin = 5
#     bins = np.arange(-4, 2, 0.5)
#     ind = np.digitize(FeH, bins)
#     ym = [len(np.where(MgFe[ind == i] < 0.4)[0]) / len(MgFe[ind == i]) for i in range(1, len(bins)) if
#           len(FeH[ind == i]) > num_min_per_bin]
#     num_per_bin = [len(MgFe[ind == i]) for i in range(1, len(bins)) if
#           len(FeH[ind == i]) > num_min_per_bin]
#
#     yerror = ym / np.sqrt(num_per_bin)
#
#     xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
#     xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
#     xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
#
#     plt.errorbar(xm, ym, yerr=yerror, xerr=[xlowe, xhighe], lw=0.5, fmt='o', markeredgecolor='black', markerfacecolor='mistyrose',
#              ecolor='grey', ms=4, markeredgewidth=0.25, label='MW-Star clusters', zorder=0)


def plot_Venn2020_scatter():

    FeH, MgFe = read_Venn2020()
    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [len(np.where(MgFe[ind == i] < 0.4)[0]) / len(MgFe[ind == i]) for i in range(1, len(bins)) if
          len(FeH[ind == i]) > num_min_per_bin]
    num_per_bin = [len(MgFe[ind == i]) for i in range(1, len(bins)) if
          len(FeH[ind == i]) > num_min_per_bin]

    yerror = ym / np.sqrt(num_per_bin)

    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerror, xerr=[xlowe, xhighe], lw=0.5, fmt='>', markerfacecolor='lavender', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.3, label='Venn et al. (2020)', zorder=0)


def plot_Yong2013_scatter():

    FeH, MgFe = read_Yong2013()
    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [len(np.where(MgFe[ind == i] < 0.4)[0]) / len(MgFe[ind == i]) for i in range(1, len(bins)) if
          len(FeH[ind == i]) > num_min_per_bin]
    num_per_bin = [len(MgFe[ind == i]) for i in range(1, len(bins)) if
          len(FeH[ind == i]) > num_min_per_bin]

    yerror = ym / np.sqrt(num_per_bin)

    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerror, xerr=[xlowe, xhighe], lw=0.5, fmt='v', markerfacecolor='antiquewhite', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.3, label='Yong et al. (2013)', zorder=0)


def plot_Venn2020():

    FeH, MgFe = read_Venn2020()
    plt.plot(FeH, MgFe, '>', markeredgecolor='black', markerfacecolor='mistyrose',
             ms=4, markeredgewidth=0.2, label='Venn et al. (2020)', zorder=-10)


def plot_Yong2013():

    FeH, MgFe = read_Yong2013()
    plt.plot(FeH, MgFe, 'v', markeredgecolor='black', markerfacecolor='antiquewhite',
             ms=4, markeredgewidth=0.2, label='Yong et al. (2013)', zorder=-10)

def read_Venn2020():
    # tabulate/compute the same ratios from Grevesse & Nobels (1993) Z/X = 0.0245
    Fe_over_H_G93 = 7.45  # Assuming 6.5% mass of Fe in Z
    Mg_over_H_G93 = 7.71  # Assuming 5.2% mass of Mg in Z
    Mg_over_Fe_G93 = Mg_over_H_G93 - Fe_over_H_G93

    # tabulate/compute the same ratios from Asplund (2009)
    Mg_Fe_Sun_Asplund = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund

    correction = Mg_over_Fe_G93 - Mg_Fe_Sun_Asplund

    input_filename = "./plotter/Observational_data/Venn_2020.txt"
    raw = np.loadtxt(input_filename, usecols=[1, 3])
    FeH = raw[:,0]
    MgFe = raw[:,1]
    no_nans = np.where((MgFe>-10) & (FeH>-10))[0]
    FeH = raw[no_nans,0]
    MgFe = raw[no_nans,1]

    MgFe += correction # Apply correction to convert to Asplund+

    return FeH, MgFe

def plot_Venn2020_avg():

    FeH, MgFe = read_Venn2020()
    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [np.mean(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    # yerr = [np.std(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    yerr = [1.96 * np.std(MgFe[ind == i]) / np.sqrt(len(MgFe[ind == i])) for i in range(1, len(bins)) if
            len(FeH[ind == i]) > num_min_per_bin]
    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerr, xerr=[xlowe, xhighe], lw=0.5, fmt='>', markerfacecolor='mistyrose', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.4, label='Venn et al. (2020)', zorder=0)


def read_Yong2013():

    input_filename = "./plotter/Observational_data/Yong_2013.txt"
    raw = np.loadtxt(input_filename, usecols=[4])
    MgFe = raw[:]

    input_filename = "./plotter/Observational_data/Yong_2013_FeH.txt"
    raw = np.loadtxt(input_filename, usecols=[11])
    FeH = raw[:]

    no_nans = np.where((MgFe>-10) & (FeH>-10))[0]
    FeH = FeH[no_nans]
    MgFe = MgFe[no_nans]

    # No need to apply correction, Yong et al. + assume Asplund+09 solar abundances

    return FeH, MgFe

def plot_Yong2013_avg():

    FeH, MgFe = read_Yong2013()
    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [np.mean(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    # yerr = [np.std(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    yerr = [1.96 * np.std(MgFe[ind == i]) / np.sqrt(len(MgFe[ind == i])) for i in range(1, len(bins)) if
            len(FeH[ind == i]) > num_min_per_bin]
    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerr, xerr=[xlowe, xhighe], lw=0.5, fmt='v', markerfacecolor='antiquewhite', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.4, label='Yong et al. (2013)', zorder=0)

def read_Venn04():

    file = './plotter/Observational_data/Venn_2004_MW_compilation.txt'
    data = np.loadtxt(file, usecols=(1, 2, 3))
    FeH = data[:, 0]
    MgFe = data[:, 1]
    Reference = data[:, 2]

    # We need to apply correction factors depending on the study:

    #compute COLIBRE standard ratios
    Mg_over_Fe_Asplund09 = constants.Mg_H_Sun_Asplund - constants.Fe_H_Sun_Asplund

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.51  # 7.67?
    Mg_over_H_AG89 = 7.58
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89
    correction_MgFe = Mg_over_Fe_AG89 - Mg_over_Fe_Asplund09
    correction_FeH = Fe_over_H_AG89 - constants.Fe_H_Sun_Asplund

    select = np.where(np.isin(Reference, [1, 2, 5, 10, 11, 12]))[0]
    FeH[select] = FeH[select] + correction_FeH
    MgFe[select] = MgFe[select] + correction_MgFe

    # tabulate/compute the same ratios from Grevesse (1996)
    Fe_over_H_Gr96 = 7.50
    Mg_over_H_Gr96 = 7.58
    Mg_over_Fe_Gr96 = Mg_over_H_Gr96 - Fe_over_H_Gr96
    correction_MgFe = Mg_over_Fe_Gr96 - Mg_over_Fe_Asplund09
    correction_FeH = Fe_over_H_Gr96 - constants.Fe_H_Sun_Asplund

    select = np.where(Reference == 6)[0]
    FeH[select] = FeH[select] + correction_FeH
    MgFe[select] = MgFe[select] + correction_MgFe

    # Correction Bensy+
    Fe_over_H = 7.45 # 7.56?
    Mg_over_H = 7.58
    Mg_over_Fe = Mg_over_H - Fe_over_H
    correction_MgFe = Mg_over_Fe - Mg_over_Fe_Asplund09
    correction_FeH = Fe_over_H - constants.Fe_H_Sun_Asplund

    select = np.where(Reference == 3)[0]
    FeH[select] = FeH[select] + correction_FeH
    MgFe[select] = MgFe[select] + correction_MgFe

    # Correction Reddy+
    Fe_over_H = 7.45
    Mg_over_H = 7.58
    Mg_over_Fe = Mg_over_H - Fe_over_H
    correction_MgFe = Mg_over_Fe - Mg_over_Fe_Asplund09
    correction_FeH = Fe_over_H - constants.Fe_H_Sun_Asplund

    select = np.where(Reference == 7)[0]
    FeH[select] = FeH[select] + correction_FeH
    MgFe[select] = MgFe[select] + correction_MgFe

    # Correction Gratton & Sneden
    Fe_over_H = 7.70
    Mg_over_H = 7.66
    Mg_over_Fe = Mg_over_H - Fe_over_H
    correction_MgFe = Mg_over_Fe - Mg_over_Fe_Asplund09
    correction_FeH = Fe_over_H - constants.Fe_H_Sun_Asplund

    select = np.where(np.isin(Reference, [14, 15]))[0]
    FeH[select] = FeH[select] + correction_FeH
    MgFe[select] = MgFe[select] + correction_MgFe

    return FeH, MgFe


def plot_Venn2004():

    FeH, MgFe = read_Venn04()
    plt.plot(FeH, MgFe, 'D', ms=2.5, markeredgewidth=0.2,
             color='lightgrey', markeredgecolor='black', label='MW-data compilation (Venn et al. 2004)', zorder=0)


def plot_Venn2004_avg():

    FeH, MgFe = read_Venn04()

    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [np.mean(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    # yerr = [np.std(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    yerr = [1.96 * np.std(MgFe[ind == i]) / np.sqrt(len(MgFe[ind == i])) for i in range(1, len(bins)) if
            len(FeH[ind == i]) > num_min_per_bin]
    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerr, xerr=[xlowe, xhighe], lw=0.5, fmt='D', markeredgecolor='black', markerfacecolor='lightgrey',
             ecolor='grey', ms=2.5, markeredgewidth=0.4, label='MW-data compilation (Venn et al. 2004)', zorder=0)


def read_Pristine():
    input_filename = "./plotter/Observational_data/Pristine_survey.txt"
    raw = np.loadtxt(input_filename, usecols=[1, 10])
    FeH = raw[:, 0]
    MgFe = raw[:, 1]
    no_nans = np.where((MgFe > -10) & (FeH > -10))[0]
    FeH = raw[no_nans, 0]
    MgFe = raw[no_nans, 1]

    # No need of correction factors; Pristine+ assume Asplund+ solar abundances
    return FeH, MgFe

def plot_Pristine_avg():

    FeH, MgFe = read_Pristine()
    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [np.mean(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    # yerr = [np.std(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    yerr = [1.96 * np.std(MgFe[ind == i]) / np.sqrt(len(MgFe[ind == i])) for i in range(1, len(bins)) if
            len(FeH[ind == i]) > num_min_per_bin]

    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerr, xerr=[xlowe, xhighe], lw=0.5, fmt='x', markerfacecolor='black', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.5, label='Pristine Survey', zorder=0)

def plot_MW_joint_datasets():

    FeH, MgFe = read_Pristine()
    x, y = read_Venn04()
    FeH = np.append(FeH, x)
    MgFe = np.append(MgFe, y)
    x, y = read_Yong2013()
    FeH = np.append(FeH, x)
    MgFe = np.append(MgFe, y)
    x, y = read_Venn2020()
    FeH = np.append(FeH, x)
    MgFe = np.append(MgFe, y)
    # _, _, x, y = read_APOGEE(("Fe_H","Mg_Fe"))
    # FeH = np.append(FeH, x)
    # MgFe = np.append(MgFe, y)

    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.5)
    ind = np.digitize(FeH, bins)
    ym = [np.mean(MgFe[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    yerr = [1.96 * np.std(MgFe[ind == i])/np.sqrt(len(MgFe[ind == i])) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    xm = [np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xlowe = [np.median(FeH[ind == i])-np.percentile(FeH[ind == i],16) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]
    xhighe = [np.percentile(FeH[ind == i],84)-np.median(FeH[ind == i]) for i in range(1, len(bins)) if len(FeH[ind == i]) > num_min_per_bin]

    plt.errorbar(xm, ym, yerr=yerr, xerr=[xlowe, xhighe], lw=0.5, fmt='o', markerfacecolor='pink', markeredgecolor='black',
             ecolor='grey', ms=4, markeredgewidth=0.5, zorder=0)
