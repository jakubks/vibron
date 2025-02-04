import numpy as np
import math
import scipy

def lambda_inner(vib_modes, HR_parameters):
    """
    Calculates the inner-sphere reorganization energy from the HR parameters
    """

    lambda_i = 0

    for i in range(len(vib_modes)):

        lambda_i += vib_modes[i] * HR_parameters[i]

    return lambda_i

def lambda_SD(freq, spec_dens):
    """
    Calculates the reorganization energy from the provided spectral density
    """

    freq, spec_dens = recast_sd(freq, spec_dens)

    lambd = scipy.integrate.simps(spec_dens/freq,freq)

    return lambd

def recast_sd(freq, spec_dens):

    if freq[0] < 0 or freq[0] > 1e-6:
        raise Exception("Invalid frequency range. The frequency range should begin between 0 and 1e-6.")
    elif spec_dens[-1] > 1e-5:
        raise Exception("Too short a range or an unphysical SD.")
    elif freq[0] == 0 and freq[1] <= 1e-6:
        freq = freq[1:]
        spec_dens = spec_dens[1:]
    elif freq[0] == 0:
        freq[0] = 1e-10
        spec_dens[0] = spec_dens[0] - 1e-10 * (spec_dens[0] - spec_dens[1])/freq[1]

    return freq, spec_dens
