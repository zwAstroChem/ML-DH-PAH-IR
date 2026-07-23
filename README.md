# ML-DH-PAH-IR
Code for High-Throughput Computation of Infrared Spectra of Polycyclic Aromatic Hydrocarbons Using a Machine-Learning-Based Double-Harmonic Model

This repository provides the source code for high-throughput computation of infrared (IR) spectra of polycyclic aromatic hydrocarbons (PAHs) using a machine-learning-based double-harmonic (ML-DH) framework. The framework integrates two complementary machine learning models:
1. Neural Network Force Field (NNFF) [DOI: 10.1021/acs.jcim.1c01380]: constructs the potential energy surface.
2. Electron Passing Neural Network (EPNN) [DOI: 10.1021/acs.jcim.0c01071]: predicts the molecular dipole moment.

Within the double-harmonic (DH) approximation, the IR spectrum of each molecule is determined at its equilibrium geometry from two sets of quantities:
1. Harmonic vibrational frequencies and normal modes, derived from the second derivatives of the NNFF-predicted potential energy surface;
2. IR intensities, obtained from the dipole derivatives predicted by the EPNN.

The NNFF and EPNN models are charge-state-specific, with separate models trained for neutral, cationic, and anionic PAHs. All models are provided in a fully trained state and can be used directly without additional training.

# Installation
1. Create and activate a virtual environment
conda create -n MLMD_env python=3.10.13 # Create a virtual environment
conda activate MLMD_env # Activate the virtual environment

2. Install the required packages:
Navigate to the respective directories and install each package in editable mode:
# Install the learned_optimization optimizer
cd lib/learned_optimization
pip install -e .
# Install NNFF
cd lib/bessel-nn-potentials-velo
pip install -e .
# Install EPNN
cd lib/epnn-main
pip install -e .

3. Install additional dependencies
pip install flax==0.7.4
pip install optax==0.1.7
pip install orbax-checkpoint==0.4.1
pip install numpy==1.26.1
pip install chex==0.1.84
pip install oryx==0.2.7
pip install scipy==1.11.3
Note: You may encounter version incompatibility warnings, which can be safely ignored.

4. Install JAX and JAXLib (CPU version)
pip install jax[cpu]==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html
Important: After installation, verify that both JAX and JAXLib are at version 0.4.19 using pip list. If not, repeat Step 4.

# Unzip the library
The required dependencies are bundled as `lib.zip`. Please unzip this file in the root directory of the repository. This will create a `./lib` folder containing all necessary library files.

# Running the Program
1. Data Preparation:
Place the .xyz files for the molecules you wish to compute in the ./inputs/XYZ/ directory. It should be in standard XYZ format. 
File format: 
Line 1: Total number of atoms.
Line 2: Comment line containing molecular metadata, MUST including the molecular charge state (Charge=0, Charge=1, or Charge=-1).
Lines 3 and beyond: Each line contains the atom type (e.g., C for carbon) and its Cartesian coordinates (x, y, z) in Angstrom.
e.g.:
18
uid=330 Charge=0
C 0.000000 0.000000 0.717038
C 0.000000 0.000000 -0.717038
C 0.000000 1.244792 -1.402729
C 0.000000 -1.244792 -1.402729
C 0.000000 1.244792 1.402729
C 0.000000 -1.244792 1.402729
C 0.000000 2.433639 -0.708550
C 0.000000 -2.433639 -0.708550
C 0.000000 2.433639 0.708550
C 0.000000 -2.433639 0.708550
H 0.000000 3.378639 -1.245348
H 0.000000 -3.378639 -1.245348
H 0.000000 3.378639 1.245348
H 0.000000 -3.378639 1.245348
H 0.000000 1.241961 -2.490506
H 0.000000 -1.241961 -2.490506
H 0.000000 1.241961 2.490506
H 0.000000 -1.241961 2.490506

2. ML-DH Calculations
Navigate to the code/ directory and execute:
python calc_harm_IR.py
The script processes all .xyz files in the input directory sequentially.

# Output Files
The computed IR spectra are saved in the ./outputs/IR/ directory. Each output file retains the same name as the corresponding input file (e.g., C10H8_330.xyz).
Runtime: The calculation for a single molecule typically takes approximately 9 minutes on a single core, although the exact runtime may vary depending on molecular size and hardware configuration.

Output file structure:
Each output .xyz file follows the structure below:
Line 1: Total number of atoms.
Line 2: Comment line containing molecular metadata, including the charge state.
Lines 3 to before spectral header: Optimized equilibrium geometry, with atomic symbols and Cartesian coordinates.
Spectral header: scaled_Freq(cm^-1) Inten(KM/mol) scale_factor
Following lines: IR spectral data in three columns:
1. Scaled harmonic vibrational frequency (cm^-1);
2. IR intensity (KM/mol);
3. Frequency scaling factor.

# Limitations
The current version is designed exclusively for pure hydrocarbon (C/H) PAHs and does not support predictions for PAH derivatives containing heteroatoms (e.g., N, O, S, etc.).
The model is based on the double-harmonic approximation and does not include anharmonic effects.

# Citation
If you use this code in your research, please cite the following article: Mai, X., & Wang, Z. (2026). A Machine-Learning-Driven Dataset of 140,000 PAH Infrared Spectra for Interpreting Aromatic Infrared Bands.

#License
This code is distributed under the Apache License 2.0, allowing for free use, modification, and distribution, including for commercial purposes, provided that proper attribution is given and the original license is included.
