import numpy as np
import scipy
from scipy.fftpack import fft, fftshift, fftfreq
from vibron.utils import vibcor, molvibs, units, const


class DonorAcceptor:
    """
    DonorAcceptor class for calculating rates of electron/energy transfer.
    All rates are calculated in s**(-1).

    Parameters:
    deltaE : energy difference between the products and reactants (i.e., the
             driving force for the reaction) (eV)
    h_DA: electronic coupling between the reactants and products
    lambda_o : outer-sphere reorganization energy (eV)
    w_cut : cut-off frequencies for the outer-sphere interactions (eV)
            (optional, only used for non-Marcus descriptions of the environment)
    temp_K : temperature (K); 300 K by default
    vib_modes_D, vib_modes_A : list of vibr. modes for the donor/acceptor (eV)
    hr_parameters_D, hr_parameters_A : list of Huang-Rhys parameters for the
                         vibrational modes in the donor/acceptor (dimensionless)
    """

    def __init__(self, deltaE = None, h_DA = 0, lambda_o = 0, w_cut = None, temp_K = 300,
       vib_modes_D = [], hr_parameters_D = [], vib_modes_A = [], hr_parameters_A = []):

        self.deltaE = deltaE
        self.h_DA= h_DA
        self.lambda_o = lambda_o
        self.w_cut = w_cut
        self.temp_K = temp_K
        self.vib_modes_D = vib_modes_D
        self.vib_modes_A = vib_modes_A
        self.hr_parameters_D = hr_parameters_D
        self.hr_parameters_A = hr_parameters_A


    def inner_reorganization(self, moiety = 'Total'):
        """
        Computes the intra-molecular reorganization energy from the HR
        parameters. Useful to verify the accuracy of the HR calculation by
        comparing it to the two-point reorganization energy a la Nelsen (1987).

        Depending on the 'moiety' keyword, it calculates it for the donor,
        the acceptor, or both moities together.

        """
        if moiety.lower() not in ['total', 'donor', 'acceptor']:
            raise Exception("Unknown moiety key. 'Total', 'Donor, and 'Acceptor' keys can be used.")

        lambda_i = 0

        if moiety.lower() == 'total' or moiety.lower() == 'donor':

            lambda_i += molvibs.lambda_inner(self.vib_modes_D, self.hr_parameters_D)

        if moiety.lower() == 'total' or moiety.lower() == 'acceptor':

            lambda_i += molvibs.lambda_inner(self.vib_modes_A, self.hr_parameters_A)


        return lambda_i

    def marcus(self, print_lambda = False):
        """
        Calculates the transfer rate according to the non-adiabatic Marcus
        theory at the specified temperature. The reorganization energy is the
        sum of the inner- and outer-sphere donor and acceptor reorganization
        energies
        """

        lambda_i = self.inner_reorganization(moiety = 'Total')
        lambda_total = lambda_i + self.lambda_o

        if lambda_total == 0:
            raise Exception('The total reorganization energy is equal to zero.')

        if print_lambda:
            if lambda_i > 0:
                print("Inner-sphere reorganization energy incorporated into the outer-sphere reorganization.")
            print(f'Total reorganization energy: {lambda_total:.3f} eV')

        kbT = self.temp_K * units.K2eV
        exponent = np.exp(- (self.deltaE + lambda_total)**2 / (4 * lambda_total * kbT ))
        pre_exponent = np.sqrt(np.pi/(lambda_total *kbT))
        rate = np.abs(self.h_DA)**2 * pre_exponent * exponent

        return rate / const.hbar

    def mlj(self, mrange = 100):
        """
        Calculates the transfer rate accoring to the single-mode and medium-
        temperature version of the Marcus-Levich-Jortner theory [Equation 3(c)
        in Jortner J. Chem. Phys., 64, 12 (1976)]:
        It assumes a single effective mode with a frequency omega >> kbT (so
        that the molecular mode is always found in its ground-state). The sum
        over the vibrational levels runs over 'mrange'.
        """

        vib_modes = np.concatenate((np.array(self.vib_modes_D),np.array(self.vib_modes_A)))
        hr_parameters = np.concatenate((np.array(self.hr_parameters_D),np.array(self.hr_parameters_A)))

        if self.lambda_o == 0:
            raise Exception('The outer-sphere reorganization energy is equal to zero.')
        elif len(vib_modes) == 0:
            raise Exception("The molecular vibrational mode is not specified. ")

        elif len(vib_modes) == 1: #single mode; evaluating the rate analytically

            kbT = self.temp_K * units.K2eV
            S = hr_parameters[0]
            omega = vib_modes[0]
            rate = 0

            for m in range(mrange): # 100 should be enough for any reasonable HR param.

                exponent = np.exp(- (self.deltaE + self.lambda_o + m*omega)**2 / (4 * self.lambda_o * kbT ))
                rate += (S**m) * exponent / np.math.factorial(m)

            pre_exponent = np.sqrt(np.pi/(self.lambda_o *kbT))
            rate = np.abs(self.h_DA)**2 * pre_exponent * rate * np.exp(-S)

        else:
            raise Exception("More than one vibrational mode specified. Use 'mjl_multi' instead.")


        return rate / const.hbar

    def mlj_multi(self, integration = 'quad'):
        """
        Calculates the transfer rate accoring to the multi-mode version of the
        Marcus-Levich-Jortner theory. Unlike 'mlj_rate' it accounts for the
        thermal population of the molecular vibrational levels.
        The necessary Fourier transform is performed numerically using either
        the quadrature ('quad') or trapezoidal integration ('trapz').
        """

        vib_modes = np.concatenate((np.array(self.vib_modes_D),np.array(self.vib_modes_A)))
        hr_parameters = np.concatenate((np.array(self.hr_parameters_D),np.array(self.hr_parameters_A)))

        if self.lambda_o == 0:
            raise Exception('The outer-sphere reorganization energy is equal to zero.')
        elif len(vib_modes) == 0:
            raise Exception("The molecular vibrational modes are not specified. ")

        if integration == 'quad':

            def corr_func(x):
                vcf = vibcor.vcfunc(vib_modes, hr_parameters, x, self.temp_K)
                ecf = vibcor.marcusfunc(self.lambda_o, x, self.temp_K)
                pcf = vcf * ecf #total phonon-vibrational correlation fun.

                return np.real(np.exp(-1j * self.deltaE * x ) * pcf)

            rate = 2 * np.abs(self.h_DA)**2 * scipy.integrate.quad(corr_func, 0, np.inf)[0]

        #elif integration == 'quad2':
            #def corr_func(x,vib_modes,hr_parameters,temp_K,lambda_o,deltaE):
                #vcf = vibcor.vcfunc(vib_modes, hr_parameters, x, temp_K)
                #ecf = vibcor.marcusfunc(lambda_o, x, temp_K)
                #pcf = vcf * ecf #total phonon-vibrational correlation fun.
                #return np.real(np.exp(-1j * deltaE * x ) * pcf)
            #rate = 2 * np.abs(self.h_DA)**2 * scipy.integrate.quad(corr_func, 0, np.inf,args=(self.vib_modes,self.hr_parameters,self.temp_K,self.lambda_o,self.deltaE))[0]

        elif integration == 'trapz':
        #Trapezoidal integration implementation:
            t_size, upper_t = vibcor.time_grids('fine')
            time = np.linspace(0, upper_t, num=t_size)
            vcf = vibcor.vcfunc(vib_modes, hr_parameters, time, self.temp_K)
            ecf = vibcor.marcusfunc(self.lambda_o, time, self.temp_K)
            pcf = vcf * ecf
            corr_func = np.exp(-1j * self.deltaE * time ) * pcf
            rate = 2 * np.abs(self.h_DA)**2 * np.real(scipy.integrate.trapz(corr_func, x=time))

        else:
            raise Exception("Integration mode not implemented. Try 'quad' or 'trapz'.")


        return rate / const.hbar

    def fgr_ohmic(self, integration = 'trapz'):
        """
        Calculates the transfer rate according to the full Fermi golden rule
        expression: both the outer-sphere environment and the molecular
        vibrational modes are treated quantum-mechanically.
        The necessary Fourier transform is performed numerically.
        """
        lambda_i = self.inner_reorganization(moiety = 'Total')
        lambda_total = lambda_i + self.lambda_o

        if lambda_total == 0:
            raise Exception('The total reorganization energy is equal to zero.')
        elif self.w_cut == None:
            raise Exception("The cut-off frequency is necessary.")

        vib_modes = np.concatenate((np.array(self.vib_modes_D),np.array(self.vib_modes_A)))
        hr_parameters = np.concatenate((np.array(self.hr_parameters_D),np.array(self.hr_parameters_A)))


        if integration == 'trapz':

            t_size, upper_t = vibcor.time_grids('coarse')
            time = np.linspace(0, upper_t, num=t_size)

            vcf = vibcor.vcfunc(vib_modes, hr_parameters, time, self.temp_K)
            ecf = vibcor.ohmicfunc(self.lambda_o, self.w_cut, time, self.temp_K)
            pcf = vcf * ecf
            corr_func = np.real(np.exp(-1j * self.deltaE * time ) * pcf)
            rate = 2 * np.abs(self.h_DA)**2 * scipy.integrate.trapz(corr_func, x=time)

        elif integration == 'quad':

            def corr_func(x, vib_modes, hr_parameters, temp_K, lambda_o, w_cut, deltaE):

                vcf = vibcor.vcfunc(vib_modes, hr_parameters, x, temp_K)
                ecf = vibcor.bathfunc_quad(lambda_o, w_cut, x, temp_K, environment)
                pcf = vcf * ecf #total phonon-vibrational correlation fun.

                return np.real(np.exp(-1j * deltaE * x ) * pcf)

                rate = 2 * np.abs(self.h_DA)**2 * scipy.integrate.quad(corr_func, 0, np.inf,
                args=(vib_modes,hr_parameters,self.temp_K,self.lambda_o,self.w_cut,self.deltaE))[0]


        return rate / const.hbar

    def fgr_custom(self, freq, SD, grid='coarse'):
        """
        Calculates electron transfer rate according to the full Fermi golden
        rule expression using a custom spectral density.
        Ignores lambda_o, w_cut, vib_modes and hr_parameters parameters.
        Evaluates all integrals numerically.
        """

        t_size, upper_t = vibcor.time_grids(grid)
        time = np.linspace(0, upper_t, num=t_size)

        pcf = vibcor.customfunc(freq, SD, time, self.temp_K)
        corr_func = np.real(np.exp(-1j * self.deltaE * time ) * pcf)
        rate = 2 * np.abs(self.h_DA)**2 * scipy.integrate.simps(corr_func, x=time)

        if self.lambda_o > 0 or len(self.vib_modes_A) > 0 or len(self.vib_modes_D) > 0:
            print("Custom spectral density provided. Ignoring vibrational/environment parameters.")

        return rate / const.hbar
