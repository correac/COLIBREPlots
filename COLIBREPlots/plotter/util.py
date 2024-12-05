import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import simpson
from simulation.utilities.constants import Mg_Fe_Sun

def plot_median_relation(x, y, color, output_name):
    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.2)
    ind = np.digitize(x, bins)

    ylo3 = [np.percentile(y[ind == i], 1) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi3 = [np.percentile(y[ind == i], 99) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ylo2 = [np.percentile(y[ind == i], 2.5) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi2 = [np.percentile(y[ind == i], 97.5) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]

    plt.fill_between(xm, ylo3, yhi3, color=color, alpha=0.2, edgecolor=None, zorder=0)
    plt.fill_between(xm, ylo2, yhi2, color=color, alpha=0.3, edgecolor=None, zorder=0)
    plt.fill_between(xm, ylo, yhi, color=color, alpha=0.5, edgecolor=None, zorder=0)
    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=100)
    plt.plot(xm, ym, '-', lw=1.5, color=color, label=output_name, zorder=100)

    select = np.where((x>-0.1)&(x<0.1))[0]
    print(np.median(x[select]),np.median(y[select]))

def plot_median_relation_one_sigma(x, y, color, output_name):
    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ylo, '-', lw=0.3, color=color, zorder=100)
    plt.plot(xm, yhi, '-', lw=0.3, color=color, zorder=100)
    plt.fill_between(xm, ylo, yhi, color=color, alpha=0.3, edgecolor=None, zorder=0)
    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=100)
    if output_name is not None:
        plt.plot(xm, ym, '-', lw=1.5, color=color, label=output_name, zorder=100)
    else:
        plt.plot(xm, ym, '-', lw=1.5, color=color, zorder=100)

def plot_median_relation_one_sigma_option(x, y, color, output_name, option):

    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ylo = [np.percentile(y[ind == i], 16) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    yhi = [np.percentile(y[ind == i], 84) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    ym = [np.median(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    if option == 1:
        plt.plot(xm, ylo, '-', lw=0.3, color=color, zorder=100)
        plt.plot(xm, yhi, '-', lw=0.3, color=color, zorder=100)
        plt.fill_between(xm, ylo, yhi, color=color, alpha=0.3, edgecolor=None, zorder=0)

    plt.plot(xm, ym, '-', lw=2.5, color='white', zorder=100)
    if output_name is not None:
        plt.plot(xm, ym, '-', lw=1.5, color=color, label=output_name, zorder=100)
    else:
        plt.plot(xm, ym, '-', lw=1.5, color=color, zorder=100)

def func_scatter(x, m):

    m_total = np.sum(m)
    mean = np.mean(x)
    sigma = m * (x-mean)**2
    sigma = np.sum(sigma)
    sigma /= m_total
    sigma = np.sqrt(sigma)
    sigma = np.std(x)
    return sigma

def func_scatter_minus_noise(x):

    y = func_scatter(x)
    y = y**2 - np.std(x)**2/len(x)
    y = np.sqrt(y)
    # y = np.percentile(x,84) - np.percentile(x,16)
    return y

def plot_scatter(x, y, m, color, output_name, linetype):

    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ym = [func_scatter(y[ind == i], m[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ym, linetype, lw=1.5, color=color, label=output_name, zorder=100)

def initial_mass_function(m):
    """
    Chabrier IMF!
    Calculate the Initial Mass Function (IMF) in linear form.

    Parameters:
    m (float): Mass for which to calculate the IMF.

    Returns:
    float: Value of the IMF at mass m. Units pc^-3
    """

    if m <= 1:
        log_m = np.log10(m)
        dm = log_m - np.log10(0.079)
        sigma = 0.69
        A = 0.852464
        xi = A * np.exp(-0.5 * dm ** 2 / sigma ** 2) / m

    else:
        A = 0.237912
        x = -2.3
        xi = A * m ** x

    xi /= np.log(10)
    return xi


def sample_IMF(num_sample):
    """
    Integrate the Initial Mass Function (IMF) over a range of masses.

    Parameters:
    m_min (float): Minimum mass to integrate over.
    m_max (float): Maximum mass to integrate over.
    num_points (int): Number of points to use in the integration. Default is 1000.

    Returns:
    float: Integrated value of the IMF over the specified mass range.
    """
    # Initial parameters
    m_min = 0.1
    m_max = 100
    num_points = 1000

    # Generate logarithmically spaced mass values for better precision
    Masses = np.logspace(np.log10(m_min), np.log10(m_max), num_points)

    # Compute the IMF values for all masses
    imf_values = np.vectorize(initial_mass_function)(Masses)

    # Integrate using the composite Simpson's rule
    # IMF_int = simpson(imf_values, x=Masses)
    IMF_int = np.sum(imf_values)
    imf_normalized = imf_values / IMF_int

    sample = np.random.choice(Masses, num_sample, p=imf_normalized)

    return sample

def plot_cumulative_function(x, y, color, output_name, linetype):

    # y += Mg_Fe_Sun
    # y = 10**y # Mg/Fe
    #
    # bins = np.arange(-4, 2, 0.25)
    # ind = np.digitize(x, bins)
    # ym = [np.sum(y[ind == i]) for i in range(1, len(bins))]
    # yf = np.cumsum(ym)
    # yf /= np.sum(ym)

    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ym = [len(np.where(y[ind == i] < 0.4)[0])/len(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ym, linetype, lw=1.5, color=color, label=output_name, zorder=-1)

def plot_mean(x, y, color, output_name, linetype):

    num_min_per_bin = 10
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ym = [np.mean(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ym, linetype, lw=1.5, color=color, label=output_name, zorder=-1)