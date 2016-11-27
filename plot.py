import matplotlib.pyplot as plt
import numpy as np
from functions import *
from matplotlib import pylab

# font management
from matplotlib.font_manager import FontProperties

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

font0 = FontProperties()
font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size(18)
legend_font = font.copy()
line_property = {"linewidth": 2.}

pylab.rcParams['figure.figsize'] = (10, 6)


def plot_gD(fig = None, axis=None):
    if not axis:
        # get the canvas
        fig, axis = plt.subplots()

    plot_zz = np.linspace(0, 5, 64)
    axis.plot(plot_zz, g(plot_zz), 'k', **line_property)
    axis.plot(plot_zz, g(plot_zz)/g(0)/(1+plot_zz), 'k--', **line_property)
    axis.legend([r"$g(z)$", r"$D(z)$"])
    axis.set_ylim(0, 1)

    # labels font
    axis.set_xlabel(r'$z$', fontproperties=font)

    # ticks font
    for tick in axis.xaxis.get_major_ticks()+axis.yaxis.get_major_ticks():
        tick.label.set_fontproperties(font)

    # legend
    axis.legend([r"$g(z)$", r"$D(z)$"],
                prop=legend_font)
    fig.savefig('gz.pdf', pad_inches=0.5, bbox_inches='tight')
    return None

# # get the canvas

def plot_Delta(fig=None, axis=None):

    if not axis:
        fig, axis = plt.subplots()

    plot_kk = np.logspace(-3, 1, 64)
    axis.loglog(plot_kk, Delta_square(plot_kk, 1), 'k', **line_property)
    axis.loglog(plot_kk, T(plot_kk), 'k--', **line_property)
    # axis.legend([r"$g(z)$", r"$D(z)$"])
    axis.set_ylim(0.001, 10)


    # labels font
    axis.set_xlabel(r'$k/h\cdot Mpc^{-1}$', fontproperties=font)
    # axis.set_ylabel(r'$\Delta^2(k, 1)$', fontproperties=font)

    # ticks font
    for tick in axis.xaxis.get_major_ticks()+axis.yaxis.get_major_ticks():
        tick.label.set_fontproperties(font)

    # legend
    axis.legend([r"$\Delta^2(k, 1)$", r"$T(k)$"],
                 prop=legend_font, loc='best')
    fig.savefig('delta.pdf', pad_inches=0.5, bbox_inches='tight')

# get the tabulated function
sig_func, table = tab_sigma()
dsdm_func = tab_dsdm(table)
cov_dis_func = tab_cov_dis(maxz=1, number=100)

def plot_sigma(fig=None, axis=None):
    if not axis:
        fig, axis = plt.subplots()

    plot_MM = np.logspace(12, 15, 64)



    axis.semilogx(plot_MM, map(sig_func, plot_MM), 'k', **line_property)
    axis.plot([M(8,0),M(8,0)], [0, 2], 'k--', **line_property)
    axis.set_ylim(0.5, 2)


    # labels font
    axis.set_xlabel(r'$M/ h^{-1}M_\odot$', fontproperties=font)
    axis.set_ylabel(r'$\sigma$', fontproperties=font, fontsize=24)

    # ticks font
    for tick in axis.xaxis.get_major_ticks()+axis.yaxis.get_major_ticks():
        tick.label.set_fontproperties(font)
    fig.savefig('sigma.pdf', pad_inches=0.5, bbox_inches='tight')
    return table


def plot_mass_func(fig= None, axis=None):

    plot_zz = np.linspace(0, 5, 50)
    fig, axis = plt.subplots()

    for MM, lp in zip([1e13, 1e14, 1e15], ["k", "k--", "k:"]):
        axis.semilogy(
            plot_zz, map(lambda z:
                         press_schechter(sig_func(MM, z),
                                         dsdm_func(MM ,z)), plot_zz),
            lp, **line_property)
    axis.set_ylim(1e-10, 1e-3)

    axis.set_xlabel(r'$z$', fontproperties=font)
    axis.set_ylabel(r'$dn/dlnM/(h^{-1}Mpc)^{-3}$', fontproperties=font)

    # ticks font
    for tick in axis.xaxis.get_major_ticks()+axis.yaxis.get_major_ticks():
        tick.label.set_fontproperties(font)

    # legend
    axis.legend([r"$M=10^{13}h^{-1}M_\odot$", r"$M=10^{14}h^{-1}M_\odot$",
                 r'$M=10^{15}h^{-1}M_\odot$'], prop=legend_font, loc='best')

    fig.savefig('mass_func.pdf', pad_inches=0.5, bbox_inches='tight')


def calculate_target_numbers():

    int_table = []
    for mm in np.linspace(14, 20, 1024):
        int_table.append( np.log10(np.e) * quad(lambda z:
                               press_schechter(sig_func(10 ** mm, z),
                                               dsdm_func(10 ** mm, z)) *
                               dvdz(z, cov_dis_func),0, 1)[0])
    tab_mass_int = interp1d(np.linspace(14, 20, 1024), int_table)
    number = quad(tab_mass_int, 14, 20)[0]
    print "total number of halo found: %g" %(number)
    return number