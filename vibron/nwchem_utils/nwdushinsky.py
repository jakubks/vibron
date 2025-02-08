import numpy as np
from scipy.spatial.transform import Rotation
from vibron.nwchem_utils import nwread
from vibron.utils import const

def align_geometries(ref_geom, fin_geom, atoms):
    """
    Aligns the geometries for the Huang-Rhys calculation
    """

    masses = atoms2masses(atoms)
    natoms = len(masses)

    masses3 = np.transpose(np.array([masses, masses, masses]))

    ref_center = np.sum(masses3*ref_geom,axis=0)
    fin_center = np.sum(masses3*fin_geom,axis=0)

    ref_geom = ref_geom - ref_center
    fin_geom = fin_geom - fin_center

    rot, rssd = Rotation.align_vectors(ref_geom, fin_geom, weights=masses)

    fin_geom_aligned = rot.apply(fin_geom)

    return ref_geom, fin_geom_aligned


def huang_rhys(ref_molecule, fin_molecule):

    """
    Calculates the Huang-Rhys parameters using the Hessian output by NWChem
    and the reference and final geometries read from NWChem files
    The calculation on the reference state must comprise geometry optimization
    followed by a frequency calculation.
    """

    ref_atoms, ref_geom = nwread.read_geometry(ref_molecule)
    fin_atoms, fin_geom = nwread.read_geometry(fin_molecule)

    if ref_atoms != fin_atoms:
        raise("Different atoms in reference and final molecules")

    natoms = len(ref_atoms)
    nmodes = natoms * 3

    hessian = nwread.read_hessian(ref_molecule)

    eigenValues, eigenVectors = np.linalg.eig(hessian)

    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    ref_aligned, fin_aligned = align_geometries(ref_geom, fin_geom, ref_atoms)

    X1 = np.ndarray.flatten(ref_aligned)
    X2 = np.ndarray.flatten(fin_aligned)
    dX = (X1-X2)

    freqs = nwread.read_frequencies(ref_molecule) / const.hbar

    masses = atoms2masses(ref_atoms) * const.amu
    M = np.sqrt(np.ndarray.flatten(np.transpose(np.array([masses, masses, masses]))))


    HR = np.zeros(nmodes)

    for p in range(nmodes):

        C_vec = eigenVectors[:,p]

        Dq = dX * M * C_vec

        HR[p] = 0.5 * freqs[p] * (np.sum(Dq))**2 / const.hbarAng


    return HR

def atoms2masses(atoms):

    masses = np.zeros(len(atoms))

    for index, atom in enumerate(atoms):

        masses[index] = const.atomic_masses[atom]


    return masses
