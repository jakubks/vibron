# Vibron

## Introduction
This package allows for the calculation of a number of molecular properties which rely on the coupling between the electronic and vibrational degrees of freedom:

### 1. Franck-Condon emission and absorption spectra

The FC spectra can be calculated using supplied Huang-Rhys parameters (`Molecule` class) or custom user-provided spectral density (`Dye` class).

### 2. Electron/energy transfer rates

A number of theoretical methods of calculating electron/energy transfer rates are implemented in the `DonorAcceptor` class. These include the non-adiabatic Marcus theory, single-mode and many-mode Marcus-Levich-Jortner theory, and the full Fermi Golden rule approach. 



The package also includes some basic functionality to read NWChem outputs and process the frequency calculation output to obtain the Huang-Rhys parameters.

Example of each usage is given in the `examples/` directory.

## Requirements

The `vibron` package is written in python3 and requires the following packages to be installed:

`numpy`

`sys`

`scipy`
