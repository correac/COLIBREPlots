import matplotlib.pylab as plt
import numpy as np


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

def func_scatter(x):

    y = np.percentile(x,84) - np.percentile(x,16)
    # y = (np.median(x)-np.percentile(x,16))**2
    # y += (np.median(x)-np.percentile(x,84))**2
    # y = np.sqrt(y)
    return y

def plot_scatter(x, y, color, output_name, linetype):

    num_min_per_bin = 5
    bins = np.arange(-4, 2, 0.25)
    ind = np.digitize(x, bins)
    ym = [func_scatter(y[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    xm = [np.median(x[ind == i]) for i in range(1, len(bins)) if len(x[ind == i]) > num_min_per_bin]
    plt.plot(xm, ym, linetype, lw=1.5, color=color, label=output_name, zorder=100)
