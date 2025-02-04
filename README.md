# Vibron

## Introduction
This package allows for the calculation of a number of molecular properties which rely on the coupling between the electronic and vibrational degrees of freedom including:

1. Franck-Condon emission and absorption spectra
  The FC spectra can be calculated using supplied Huang-Rhys parameters or custom user-provided spectral density.
2. Electron transfer rates

The package also includes some basic functionality to read NWChem outputs and process the frequency calculation output to obtain the Huang-Rhys parameters.

Example of each usage is given in the `examples/` directory.

## Requirements

The `vibron` is written in python3 and requires the following packages to be installed:

`numpy`

`sys`

`scipy`
