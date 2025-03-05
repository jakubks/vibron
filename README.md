<p align="center">
<img src="https://github.com/jakubks/vibron/blob/main/vibron_logo.png" width="348">
<p>
  
## Introduction

`vibron.py` is a Python package that allows for an easy calculation of a number of molecular properties which rely on the coupling between the electronic and vibrational degrees of freedom such as:

### 1. Franck-Condon Emission and Absorption Spectra

The FC absorption/emission spectra and fluorescence lifetimes can be calculated using provided Huang-Rhys parameters (`Molecule` class) or a custom user-provided spectral density (`Dye` class).

### 2. Charge/Energy Transfer Rates

A number of theoretical methods of calculating electron/energy transfer rates are implemented in the `DonorAcceptor` class. These include the non-adiabatic Marcus theory, single-mode and many-mode Marcus-Levich-Jortner theory, and the full Fermi Golden rule approach (which is also implemented for a custom user-provided spectral density). 

### 3. Heterogenous Electron Transfer Rates
--Work in progress--

### 4. Resonant Charge Transport through Molecular Junctions
--Work in progress--

The `vibron.py` package also includes some basic functionality to read NWChem/Gaussian outputs and process the frequency calculation outputs to obtain the Huang-Rhys parameters.

Multiple examples of usage are provided as Jupyter notebooks in the `examples/` directory.

## Requirements

The `vibron` package is written in python3 and requires the following packages to be installed:

`numpy`

`sys`

`scipy`

## Contributions

This package is still a work in progress. Questions, problems, suggestions? [Get in touch!](https://sites.google.com/view/jakubksowa#h.p_Y6ozqPgyTCZb)
