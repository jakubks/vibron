import numpy as np
from scipy.fftpack import fft, fftshift, fftfreq

from vibron.utils import vibcor, molvibs, units, const

class Molecule:
    """
    Molecule class for calculating Franck-Condon emission
    and absorption spectra with the following parameters:

    ex_energy : excitation energy (typically S_1 energy wrt. S_0) (eV)
    lambda_o : outer-sphere (environmental) reorganization energy (eV)
    w_cut : cut-off frequency for the outer-sphere coupling (eV)
            (optional; required for non-Marcus descriptions of the environment)
    vib_modes : a list of vibrational mode frequencies (eV)
    hr_parameters : a list of Huang-Rhys parameters for the vibrational modes
                    (dimensionless; the length and order has to be consistent
                    with 'vib_modes')
    gamma : lifetime associated with the transition (eV)
    dipole : transition dipole moment (atomic units, optional)
    temp_K : temperature (K); 300 K by default

    """

    def __init__(self, ex_energy = None, lambda_o = 0, w_cut = None, vib_modes = [],
           hr_parameters = [], gamma = 0, dipole = [], temp_K = 300):

        self.ex_energy = ex_energy
        self.lambda_o = lambda_o
        self.w_cut = w_cut
        self.vib_modes = vib_modes
        self.hr_parameters = hr_parameters
        self.gamma = gamma
        self.dipole = dipole
        self.temp_K  = temp_K

    def inner_reorganization(self):
        """
        Computes the intra-molecular reorganization energy from the HR
        parameters. Useful to verify the accuracy of the HR calculation by
        comparing it to the two-point reorganization energy a la Nelsen (1987).
        """

        return molvibs.lambda_inner(self.vib_modes, self.hr_parameters)

    def emission_spectrum(self, environment='Marcus', grid="fine"):
        """
        Computes the Franck-Condon emission spectrum of the molecule at the
        specified temperature.

        environment : the environmental (solvent/crystal) interactions are
                  implemented through a Marcus-type approach (keyword 'Marcus'),
                  and an Ohmic bath (keyword 'Ohmic').
                  The latter option requires the cut-off frequency argument.

        Three ('fine','xfine','coarse') integration grids for the FFT are
        implemented.
        """

        if self.lambda_o == 0 and self.gamma == 0 and len(self.vib_modes) == 0:
            raise ValueError('No parameters for the spectrum specified.')
        elif self.lambda_o == 0 and self.gamma == 0:
            print('Warning: Without outer-sphere coupling or lifetime broadening, the calculation is unstable. Introducing small gamma is recommended.')

        t_size, upper_t = time_grid(grid)
        positive = int(t_size/2)
        time = np.linspace(0, upper_t, t_size)
        energy_range = 2* np.pi * fftfreq(t_size,upper_t/t_size)[:positive]

        vcf = vibcor.vcfunc(self.vib_modes, self.hr_parameters, time, self.temp_K)

        if environment.lower() == "marcus":
            ecf = vibcor.marcusfunc(self.lambda_o, time, self.temp_K)

        elif environment.lower() == 'ohmic':
            ecf = vibcor.ohmicfunc(self.lambda_o, self.w_cut, time, self.temp_K)

        else:
            raise Exception("Environment type unknown.")

        pcf = vcf * ecf #total phonon-vibrational correlation fun.

        Corr_func = np.exp(1j * self.ex_energy * time - self.gamma * time) * pcf
        FourierT = np.real(fft(Corr_func)[:positive]) - 0.5


        if self.dipole == []:

            print('No dipole moment was given. Returning a normalized spectrum.')
            spectrum = FourierT * energy_range**3
            spectrum = spectrum / np.max(spectrum)

        else:

            spectrum = FourierT * (upper_t/t_size) * energy_range**3 * dipole2(self.dipole)

        return energy_range, spectrum



    def absorption_spectrum(self, environment='Marcus', grid="fine"):
        """
        The same as for the emission spectrum
        """

        if self.lambda_o == 0 and self.gamma == 0 and len(self.vib_modes) == 0:
            raise ValueError('No parameters for the spectrum specified.')
        elif self.lambda_o == 0 and self.gamma == 0:
            print('Warning: Without outer-sphere coupling or lifetime broadening, the calculation is unstable. Introducing small gamma is recommended.')

        t_size, upper_t = time_grid(grid)
        positive = int(t_size/2)
        time = np.linspace(0, upper_t, t_size)
        energy_range = 2* np.pi * fftfreq(t_size,upper_t/t_size)[:positive]

        vcf = vibcor.vcfunc(self.vib_modes, self.hr_parameters, time, self.temp_K)

        if environment.lower() == "marcus":
            ecf = vibcor.marcusfunc(self.lambda_o, time, self.temp_K)

        elif environment.lower() == 'ohmic':
            ecf = vibcor.ohmicfunc(self.lambda_o, self.w_cut, time, self.temp_K)

        else:
            raise Exception("Environment type unknown.")

        pcf = np.conjugate(vcf * ecf) #total phonon-vibrational correlation fun.

        Corr_func = np.exp(1j * self.ex_energy * time - self.gamma * time) * pcf
        Fourier = np.real(fft(Corr_func)[:positive]) - 0.5

        if self.dipole == []:

            print('No dipole moment was given. Returning a normalized spectrum.')
            spectrum = Fourier * energy_range
            spectrum = spectrum / np.max(spectrum)

        else:

            spectrum = FourierT * (upper_t/t_size) * energy_range * dipole2(self.dipole)

        return energy_range, spectrum

class Dye:
    """
    Dye class for calculating Franck-Condon spectra using a custom phononic
    spectra density

    Parameters:
    ex_energy : excitation energy (typically S_1 energy wrt. S_0) (eV)
    freq : frequency range for the spectral density
    J : vibrational (phononic) spectral density
    gamma : lifetime associated with the transition (eV)
    dipole : transition dipole moment (optional)
    temp_K : temperature (K); 300 K by default
    """

    def __init__(self, ex_energy = None, freq = [], spec_dens = [],
                 gamma = 0, dipole = [], temp_K = 300):

        self.ex_energy = ex_energy
        self.freq = freq
        self.spec_dens = spec_dens
        self.gamma = gamma
        self.dipole = dipole
        self.temp_K  = temp_K

    def reorganization(self):
        """
        Computes the reorganization energy from the provided spectral density.
        """

        return molvibs.lambda_SD(self.freq, self.spec_dens)

    def emission_spectrum(self, grid="fine"):
        """
        Computes the Franck-Condon emission spectrum of the dye at the
        specified temperature.

        Three ('fine','xfine','coarse') integration grids for the FFT are
        implemented.
        """

        if self.spec_dens == [] or self.freq == [] or self.ex_energy == None:
            raise ValueError('No parameters for the spectrum specified.')


        t_size, upper_t = time_grid(grid)
        positive = int(t_size/2)
        time = np.linspace(0, upper_t, t_size)
        energy_range = 2* np.pi * fftfreq(t_size,upper_t/t_size)[:positive]

        pcf = vibcor.customfunc(self.freq, self.spec_dens, time, self.temp_K)

        Corr_func = np.exp(1j * self.ex_energy * time - self.gamma * time) * pcf
        FourierT = np.real(fft(Corr_func)[:positive]) - 0.5

        if self.dipole == []:

            print('No dipole moment was given. Returning a normalized spectrum.')
            spectrum = FourierT * energy_range**3
            spectrum = spectrum / np.max(spectrum)

        else:

            spectrum = FourierT * (upper_t/t_size) * energy_range**3 * dipole2(self.dipole)

        return energy_range, spectrum

    def absorption_spectrum(self, grid="fine"):
        """
        Computes the Franck-Condon emission spectrum of the dye at the
        specified temperature.

        Three ('fine','xfine','coarse') integration grids for the FFT are
        implemented.
        """

        if self.spec_dens == [] or self.freq == [] or self.ex_energy == None:
            raise ValueError('No parameters for the spectrum specified.')

        t_size, upper_t = time_grid(grid)
        positive = int(t_size/2)
        time = np.linspace(0, upper_t, t_size)
        energy_range = 2* np.pi * fftfreq(t_size,upper_t/t_size)[:positive]

        pcf = vibcor.customfunc(self.freq, self.spec_dens, time, self.temp_K)
        pcf = np.conjugate(pcf)
        Corr_func = np.exp(1j * self.ex_energy * time - self.gamma * time) * pcf
        FourierT = np.real(fft(Corr_func)[:positive]) - 0.5

        if self.dipole == []:

            print('No dipole moment was given. Returning a normalized spectrum.')
            spectrum = FourierT * energy_range
            spectrum = spectrum / np.max(spectrum)

        else:

            spectrum = FourierT * (upper_t/t_size) * energy_range * dipole2(self.dipole)

        return energy_range, spectrum


def dipole2(dipole):
    "Checks is self.dipole has a right format and outputs its square"

    if type(dipole) is int or type(dipole) is float:
        d2 = dipole**2

    elif len(dipole) == 1 or len(dipole) == 3:
        d2 = np.linalg.norm(np.array(dipole))**2
    else:
        raise Exception("Dipole should be a scalar or a 3-d vector.")

    return d2

def time_grid(grid):
    if grid =="fine":
        t_size = 1e5
        upper_t  = 5e3
    elif grid =="xfine":
        t_size = 2e6
        upper_t = 5e4
    elif grid =="coarse":
        t_size = 1e4
        upper_t = 1e3

    return int(t_size), int(upper_t)
