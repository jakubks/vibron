import numpy as np

class Junction:
    """
    Junction class for calculating resonant current through a molecular
    junction.
    Parameters:

    epsilon : molecular energy level with respect to the Fermi level of the
    unbiased electrodes
    lambda_o : outer-sphere (environmental) reorganization energy
    w_cut : cut-off frequency for the outer-sphere coupling
            (optional; required for non-Marcus descriptions of the environment)
    vib_modes : a list of vibrational mode frequencies
    HR_parameters : a list of Huang-Rhys parameters for the vibrational modes
                    (the length and order has to be consistent with vib_modes)
    Gamma_L, Gamma_R : strength of the molecule-electrode interactions
                       Gamma_i = 2 * pi * |V_i|**2 * rho_i  where V_i is the
                       electronic coupling and rho_i is the density of states


    """

    def __init__(self, epsilon = None, lambda_o = 0, w_cut = None, vib_modes = [], HR_parameters = [], Gamma = 0):

        self.epsilon = epsilon
        self.lambda_o = lambda_o
        self.w_cut = w_cut
        self.vib_modes = vib_modes
        self.HR_parameters = HR_parameters
        self.Gamma = Gamma

    def inner_reorganization(self):
        """
        Computes the intra-molecular reorganization energy from the HR
        parameters. Useful to verify the accuracy of the HR calculation by
        comparing it to the two-point reorganization energy a la Nelsen (1987).
        """

        return molvibs.lambda_inner(self.vib_modes, self.HR_parameters)
