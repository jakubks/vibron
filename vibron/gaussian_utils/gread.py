import numpy as np
from vibron.utils import const, units


def read_natoms(nwfile):

    with open(nwfile) as file:

        f = file.readlines()

        for line in f:
            if ' NAtoms=' in line:
                natoms = int(line.split()[1])

    return natoms


def read_frequencies(file_path):
    """Extracts vibrational frequencies from a Gaussian output file."""
    frequencies = []
    with open(file_path, 'r') as f:
        for line in f:
            if "Frequencies ---" in line:
                freqs = [float(x) for x in line.split()[2:]]  # Extract frequency values
                frequencies.extend(freqs)
    return np.array(frequencies) * units.wavenumber2eV


def extract_vibrational_vectors(file_path, natoms):
    """Extracts vibrational displacement vectors from a Gaussian output file."""
    eigen = np.zeros((natoms*3,natoms*3-6))
    kk = 0
    with open(file_path, 'r') as f:

        file = f.readlines()
        for jk in range(len(file)):
            if " Coord Atom Element:" in file[jk]:# and "X" in line and "Y" in line and "Z" in line:
                jj = 0
                for kl in range(jk+1,jk+natoms*3+1):
                    linee = file[kl][20:].split()
                    numbs = [float(x) for x in linee]
                    for q in range(len(numbs)):
                        eigen[jj,kk+q] = numbs[q]
                    jj += 1
                kk +=5
    return eigen


def extract_geometry(logfile, natoms):
    """
    Extracts the optimized geometry from a Gaussian output file and returns it as a NumPy array.

    Parameters:
        logfile (str): Path to the Gaussian output file.

    Returns:
        np.ndarray: Optimized geometry as an (N, 4) array, where N is the number of atoms.
                    Columns: [Atomic number, x, y, z]
    """
    with open(logfile, 'r') as file:
        lines = file.readlines()

    # Locate the final optimized geometry
    geom_start = None
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:# or 'Input orientation:' in line:
            geom_start = i

    if geom_start is None:
        raise ValueError("Optimized geometry not found in the Gaussian output file.")

    # Extract atomic coordinates (after last 'Standard orientation:')
    geom_data = np.zeros((natoms,3))
    atoms = []

    for ij in range(geom_start+5,geom_start+5+natoms):
        line = lines[ij].split()
        geom_data[ij-geom_start-5,0] = float(line[3])
        geom_data[ij-geom_start-5,1] = float(line[4])
        geom_data[ij-geom_start-5,2] = float(line[5])

        if line[1] == '1':
            atoms.append('H')
        elif line[1] == '6':
            atoms.append('C')
        elif line[1] == '14':
            atoms.append('Si')
        else:
            raise Exception("atoms")

    return atoms, geom_data

def read_energy(nwfile):
    """
    Reads the energy of an SCF calculation
    Returns energy in eV
    """

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'SCF Done =' in f[jk]:
                energy = float(f[jk].split()[5])

    return energy * units.Ha2eV

def read_tddft(nwfile):
    """
    Reads the energy of a TD-DFT calculation
    Returns energy in eV
    """

    with open(nwfile) as file:

        f = file.readlines()

        for jk in range(len(f)):
            if 'Total Energy, E(TD-' in f[jk]:
                energy = float(f[jk].split()[4])

    return energy * units.Ha2eV
