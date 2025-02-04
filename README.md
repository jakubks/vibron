# vibron.py

![alt text](https://github.com/jakubks/vibron/logo.jpg?raw=true)

`vibron.py` is a Python package that allows for an easy calculation of a number of molecular properties which rely on the coupling between the electronic and vibrational degrees of freedom such as:

### 1. Franck-Condon emission and absorption spectra

The FC spectra can be calculated using supplied Huang-Rhys parameters (`Molecule` class) or a custom user-provided spectral density (`Dye` class).

### 2. Electron/energy transfer rates

A number of theoretical methods of calculating electron/energy transfer rates are implemented in the `DonorAcceptor` class. These include the non-adiabatic Marcus theory, single-mode and many-mode Marcus-Levich-Jortner theory, and the full Fermi Golden rule approach (which is also implemented for a custom user-provided spectral density). 

### 3. Heterogenous electron transfer rates
--Work in progress--

### 4. Resonant charge transport through a molecular junction
--Work in progress--

The `vibron.py` package also includes some basic functionality to read NWChem outputs and process the NWChem frequency calculation output to obtain the Huang-Rhys parameters.

Several examples of usage are provided as Jupyter notebooks in the `examples/` directory.

## Requirements

The `vibron` package is written in python3 and requires the following packages to be installed:

`numpy`

`sys`

`scipy`
