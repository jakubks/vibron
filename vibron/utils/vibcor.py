import numpy as np
import scipy
from vibron.utils import units, molvibs


def coth(x):
    return np.cosh(x) / np.sinh(x)

def marcusfunc(lambda_o, time, temp_K):
    """
    Marcus (Gaussian) correlation function
    """

    beta = 1/(temp_K * units.K2eV) # into 1/eV
    MCF = np.exp(- 1j * lambda_o * time - lambda_o * time**2 /beta)

    return MCF

def vcfunc(frequencies, HRfactors, time, temp_K):
    """
    Vibrational correlation function for discreet modes
    """

    beta = 1/(temp_K * units.K2eV) # into 1/eV

    nmodes = len(frequencies)
    VCF = np.exp(0*time)

    for ijk in range(nmodes):

        omega = frequencies[ijk]

        if omega > 1e-6: # to exclude translations/rotations

            VCF = VCF * np.exp(-HRfactors[ijk]* (coth(beta*omega/2)*(1-np.cos(omega*time)) + 1j*np.sin(omega*time)))

    return VCF

def ohmicfunc(lambda_o, w_cut, time, temp_K):
    """
    Vibrational correlation function for ohmic/super-ohmic phonon baths both
    with an exponential cut-off
    """

    if w_cut == None:
        raise ValueError("Outer-sphere cut-off frequency is missing.")

    beta = 1/(temp_K * units.K2eV) # into 1/eV
    freq = np.linspace(1e-6, w_cut*30, int(1e3))

    J = lambda_o * (freq/w_cut) * np.exp(-freq/w_cut)

    NT = len(time)
    entegral = np.zeros(NT) + 1j*np.zeros(NT)
    Coth_ = coth(beta*freq/2)
    HR_J = J / freq**2

    for jk in range(NT):
        t = time[jk]
        real = scipy.integrate.simps(HR_J * Coth_ * (np.cos(freq*t)-1),freq)
        imaginary = scipy.integrate.simps(-HR_J * np.sin(freq*t),freq)
        entegral[jk] = real + 1j*imaginary

    return np.exp(entegral)


def customfunc(freq, spectr_dens, time, temp_K):
    """
    Vibrational correlation function for a custom spectral density J(freq)
    """
    freq, spec_dens = molvibs.recast_sd(freq, spectr_dens)

    beta = 1/(temp_K * units.K2eV) # into 1/eV

    NT = len(time)
    entegral = np.zeros(NT) + 1j*np.zeros(NT)
    Coth_ = coth(beta*freq/2)
    HR_J = spec_dens / freq**2

    for jk in range(NT):
        t = time[jk]
        real = scipy.integrate.trapz(HR_J * Coth_ * (np.cos(freq*t)-1),freq)
        imaginary = scipy.integrate.trapz(-HR_J * np.sin(freq*t),freq)
        entegral[jk] = real + 1j*imaginary

    return np.exp(entegral)


def time_grids(grid):
    if grid =="fine":
        t_size = 1e5
        upper_t  = 1e3
    elif grid =="xfine":
        t_size = 1e7
        upper_t = 1e4
    elif grid =="coarse":
        t_size = 1e4
        upper_t = 1e3

    return int(t_size), int(upper_t)
