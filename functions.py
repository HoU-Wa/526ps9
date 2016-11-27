import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from __init__ import OLD_COSMO

def cosmo(OLD_COSMO):
    om = 0.266
    if OLD_COSMO:
        om = 1.
    h0 = 0.71
    if OLD_COSMO:
        h0 = 0.5
    return om, h0

om, h0 = cosmo(OLD_COSMO)
ol = 1 - om
ob = 0.02258 / h0 ** 2
n = 0.963
A = 2.43e-9
k_piv_co = 0.002 / h0  # h/Mpc
rho_cric_0 = 2.77554920617e+11  # h^2*msun/Mpc**3


def hubble_par(z):
    return np.sqrt((1 + z) ** 3 * om + ol)


def cric_dens(z):
    return hubble_par(z) ** 2


def omz(z):
    return om * (1 + z) ** 3 / cric_dens(z)


def olz(z):
    return ol / cric_dens(z)


def g(z):
    return 2.5 * omz(z) / \
           (omz(z) ** (4. / 7.) - olz(z) +
            (1 + 0.5 * omz(z)) * (1 + olz(z) / 70))


def q(k_co):
    return k_co / om / h0 / np.exp(-ob - 1.3 * ob / om)


def T(k_co):
    return np.log(1 + 2.34 * q(k_co)) / (2.34 * q(k_co)) * \
           (1 + 3.89 * q(k_co) + 16.1 ** 2 * q(k_co) ** 2 + 5.46 ** 3 *
            q(k_co) ** 3 + 6.71 ** 4 * q(k_co) ** 4) ** (-0.25)


def Delta_square(k_co, a):
    # H_0 = 100 h km/s/Mpc = (100km/s)/speed_of_light*(h/Mpc) = 1/3000 h/Mpc
    return A * 4. / 25 / om ** 2 * (k_co / k_piv_co) ** (n - 1) * (k_co / (
    1. / 3000)) ** 4 * \
           (a * g(1. / a - 1)) ** 2 * T(k_co) ** 2


def j1(x):
    return np.sin(x) / x ** 2 - np.cos(x) / x


def R(M, z):
    return (3 * M / 4. / np.pi / (om * rho_cric_0 * (1 + z) ** 3)) ** (
    1. / 3)  # h**(-1)*Mpc


def M(R, z):
    return 4. / 3 * np.pi * R ** 3 * (om * rho_cric_0 * (1 + z) ** 3)


def sigma_square_intgrand(k_co, z, M):
    rr = R(M, 0)
    return Delta_square(k_co, 1. / (1. + z)) * 9 *\
           j1(k_co * rr) ** 2 / (k_co * rr) ** 2 / k_co


def tab_sigma():
    sample = np.logspace(12, 20, 256)
    table = []

    for mm in sample:
        table.append(np.sqrt(
            quad(lambda k: sigma_square_intgrand(k, 0, mm), 0, np.inf)[0]))

    interper = interp1d(sample, table)

    def sigma(m, z=0):
        return 1 / (1. + z) * g(z) / g(0) * interper(m)

    return sigma, interper


def tab_dsdm(interper_sigma):
    sample = np.logspace(12, 20, 256)
    table_prime = []
    delta = 1e-3
    for mm in sample:
        prime = np.sqrt(
            quad(lambda k: sigma_square_intgrand(k, 0, mm * (1. + delta)), 0,
                 np.inf)[0])
        table_prime.append(abs((interper_sigma(mm) - prime) / delta / mm))

    interper = interp1d(sample, table_prime)

    def dsdm(m, z=0):
        return 1 / (1. + z) * g(z) / g(0) * interper(m)

    return dsdm


def press_schechter(sig, dsigmadm):
    return np.sqrt(2. / np.pi) * om * rho_cric_0 * 1.686 / sig ** 2 * abs(
        dsigmadm) * \
           np.exp(-1.686 ** 2 / 2. / sig ** 2)


def tab_cov_dis(maxz=5, number=100):
    sample = np.linspace(0, maxz, number)
    table = []
    for zz in sample:
        # H_0 = 100 h km/s/Mpc = (100km/s)/speed_of_light*(h/Mpc) = 1/3000 h/Mpc
        table.append(3000 * quad(lambda zz: 1. / hubble_par(zz), 0, zz)[0])
    return interp1d(sample, table)


def dvdz(z, cov_dis_func):
    # whole sky covers 41253 sqaure degrees
    # H_0 = 100 h km/s/Mpc = (100km/s)/speed_of_light*(h/Mpc) = 1/3000 h/Mpc
    return cov_dis_func(z) ** 2 * 3000 / hubble_par(z) *\
           (5000. / 41253) * 4 * np.pi