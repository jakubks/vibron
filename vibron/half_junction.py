import numpy as np

class HalfJunction:
    """HalfJunction class for calculating the rates of heterogenous electron
    transfer (metal/semiconductor <-> molecule) with the following
    parameters:

    epsilon : molecular energy level with respect to the Fermi level of the
    unbiased electrode
    lambda_o : outer-sphere (environmental) reorganization energy (eV)
    w_cut : cut-off frequency for the outer-sphere coupling (eV)
            (optional; required for non-Marcus descriptions of the environment)
    vib_modes : a list of vibrational mode frequencies (eV)
    HR_parameters : a list of Huang-Rhys parameters for the vibrational modes
                    (the length and order has to be consistent with vib_modes)
    Gamma : strength of the molecule-electrode interaction (eV)
            Gamma = 2 * pi * |V|**2 * rho
            where V is the electronic coupling and rho is the density of states
    temp_K : temperature (K); 300 K by default

    """

    def __init__(self, epsilon = None, lambda_o = 0, w_cut = None, vib_modes = [],
                HR_parameters = [], Gamma = 0, temp_K = 300):

        self.epsilon = epsilon
        self.lambda_o = lambda_o
        self.w_cut = w_cut
        self.vib_modes = vib_modes
        self.HR_parameters = HR_parameters
        self.Gamma = Gamma
        self.temp_K = temp_K

    def inner_reorganization(self):
        """
        Computes the intra-molecular reorganization energy from the HR
        parameters. Useful to verify the accuracy of the HR calculation by
        comparing it to the two-point reorganization energy a la Nelsen (1987).
        """

        return molvibs.lambda_inner(self.vib_modes, self.HR_parameters)
