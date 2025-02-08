import numpy as np
from vibron.utils import units

def read_natoms(nwfile):

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'XYZ format geometry' in f[jk]:
                natoms = int(f[jk+2])

    return natoms


def read_geometry(nwfile):

    natoms = read_natoms(nwfile)
    geometry = np.zeros((natoms,3))

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Output coordinates in angstroms' in f[jk]:

                atoms = []

                for kl in range(natoms):
                    line = f[jk+4+kl].split()

                    atoms.append(line[1])
                    geometry[kl,0] = float(line[3])
                    geometry[kl,1] = float(line[4])
                    geometry[kl,2] = float(line[5])

    return atoms, geometry

def read_frequencies(nwfile):
    """
    Reads the frequencies of vibrational modes
    Returns mode energies in eV
    """
    natoms = read_natoms(nwfile)
    freqs = np.zeros(3*natoms)

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Normal Eigenvalue' in f[jk]:

                for kl in range(natoms*3):
                    line = f[jk+3+kl].split()

                    freqs[kl] = float(line[1])

    if np.min(freqs) < -0.001:

        raise Exception("Imaginary frequencies in the output file")

    return freqs * units.wavenumber2eV


def read_hessian(nwfile):

    natoms = read_natoms(nwfile)
    nmodes = 3*natoms
    nrows = int(np.ceil(nmodes/10))
    hessian = np.zeros((nmodes,nmodes))

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'MASS-WEIGHTED PROJECTED HESSIAN' in f[jk]:

                for kl in range(nrows):

                    q = 0
                    for qq in range(kl):
                        q = q + nmodes + 4 - qq*10

                    for lm in range(nmodes - kl*10):
                        line = f[jk + 6 + lm + q].split()

                        for mn in range(np.min([lm,9]) + 1):

                            hessian[lm + 10*kl,mn + 10*kl] = float(line[mn+1].replace('D','E'))

    hessian = hessian + np.transpose(hessian) - np.diag(np.diag(hessian))

    return hessian

def read_optenergy(nwfile):
    """
    Reads the energy of the optimized structure (the nwfile has to be an
    optimization calculation)
    Returns energy in eV
    """

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Optimization converged' in f[jk]:
                energy = float(f[jk+6].split()[2])

    return energy * units.Ha2eV

def read_energy(nwfile):
    """
    Reads the energy of a single-point calculation
    Returns energy in eV
    """

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Total DFT energy =' in f[jk]:
                energy = float(f[jk].split()[4])

    return energy * units.Ha2eV

def read_tddft(nwfile, nroot = 1, mult = 'singlet'):
    """
    Reads the excited state energy of the nth-root of either the singlet
    or triplet as specified by mult = 'singlet' or 'triplet'
    Returns excited state energy in eV
    """

    line_gs = 'Ground state energy ='
    line_ex = 'Root   ' + str(nroot) + ' ' + mult

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if line_gs in f[jk]:
                gs_energy = float(f[jk].split()[4])

            elif line_ex in f[jk]:
                tddft_energy = float(f[jk].split()[4])

    return (gs_energy + tddft_energy) * units.Ha2eV

def read_transitiondipole(nwfile, nroot = 1, mult = 'singlet'):
    """
    Reads the transition dipole for the n-th excited state
    Returns the dipole moment in Debye
    """
    dipole = np.zeros(3)

    line_ex = 'Root   ' + str(nroot) + ' ' + mult

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):

            if line_ex in f[jk]:
                dipole[0] = float(f[jk+2].split()[3])
                dipole[1] = float(f[jk+2].split()[5])
                dipole[2] = float(f[jk+2].split()[7])


    return dipole * units.au2D

def read_ETcoupling(nwfile):
    """
    Reads the energy of a single-point calculation
    """

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Electron Transfer Coupling Energy' in f[jk]:
                coupling = float(f[jk+2].split()[0])

    return coupling
