#!/usr/bin/env python

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import re
import copy
import pickle
import sys
import gc
from tqdm import tqdm
import argparse

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np

from typing import Any, Tuple
import time
from functools import partial
import scipy
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

from neuralil.bessel_descriptors import (
    PowerSpectrumGenerator,
    get_max_number_of_neighbors,
)

import operator
import flax
import flax.core

from neuralil.model import (
            NeuralILModelInfo,
                NeuralILwithMorse,
                    ResNetCore,
                        update_energy_offset,
                        )
from neuralil.plain_ensembles.model import PlainEnsemblewithMorse

from neuralil.utilities import *

from neuralil.utilities import create_array_shuffler, draw_urandom_int32

from epnn.model_charge import EPNN, MLP, EPNNModelInfo
from epnn.preprocessing import create_batched_graph_list, get_init_charges
from epnn.utils import get_nodes_from_graph, split_array_train_val, split_array_equal_size

import ase
from ase.io import read
from ase import units
from neuralil.ase_integration_gai_jit_charge import NeuralILASECalculator
from ase.optimize import BFGS
   
def Hess2freq(hessian,mol_coord,mol_weight):
    # Function: Calculate the vibration frequency
    # Description: Compute harmonic vibrational frequencies and normal modes from the Hessian matrix (second derivatives of the PES)
    
    E_h = physical_constants["electron volt"][0] #1.602176634e-19 J  #NNFF unit is ev,to J
    a_0 = 1.0e-10 #physical_constants["Angstrom star"][0]  #1*10**(-10) m #NNFF unit is Angstrom,to m
    
    N_A = physical_constants["Avogadro constant"][0]   #6.02214076e+23 mol^-1
    c_0 = physical_constants["speed of light in vacuum"][0]

    mol_coord = np.array(mol_coord, dtype=np.float64)  
    mol_weight = np.array(mol_weight, dtype=np.float64)
    natm = len(mol_coord)

    mol_hess = np.array(hessian, dtype=np.float64)  #(3N,3N) unit: ev/Angstrom^2
    mol_hess = (mol_hess + mol_hess.T) / 2
    mol_hess = mol_hess.reshape((natm, 3, natm, 3))

    theta = np.einsum("AtBs, A, B -> AtBs", mol_hess, 1 / np.sqrt(mol_weight), 1 / np.sqrt(mol_weight)).reshape(3 * natm, 3 * natm)
    e, q_1 = np.linalg.eigh(theta)   #theta (3N,3N), mass weighted cartesian coordinates,e is the eigenvalue, converted to wavenumber according to formula (1)
    freq_cm_1 = np.sqrt(np.abs(e * E_h * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e > 0) * 2 - 1)
    #'q_start'  # (3N,3N),Contains translational and rotational frequencies
    #'freq_cm_1'  #Contains translational and rotational frequencies
    
    center_coord = (mol_coord * mol_weight[:, None]).sum(axis=0) / mol_weight.sum()	
    centered_coord = mol_coord - center_coord
    #centered_coord    # Coordinates of the centroid coordinate system

    rot_tmp = np.zeros((natm, 3, 3))
    rot_tmp[:, 0, 0] = centered_coord[:, 1]**2 + centered_coord[:, 2]**2
    rot_tmp[:, 1, 1] = centered_coord[:, 2]**2 + centered_coord[:, 0]**2
    rot_tmp[:, 2, 2] = centered_coord[:, 0]**2 + centered_coord[:, 1]**2
    rot_tmp[:, 0, 1] = rot_tmp[:, 1, 0] = - centered_coord[:, 0] * centered_coord[:, 1]
    rot_tmp[:, 1, 2] = rot_tmp[:, 2, 1] = - centered_coord[:, 1] * centered_coord[:, 2]
    rot_tmp[:, 2, 0] = rot_tmp[:, 0, 2] = - centered_coord[:, 2] * centered_coord[:, 0]
    rot_tmp = (rot_tmp * mol_weight[:, None, None]).sum(axis=0)
    _, rot_eig = np.linalg.eigh(rot_tmp)
    #rot_eig       #Characteristic vector of moment of inertia

    rot_coord = np.einsum("At, ts, rw -> Asrw", centered_coord, rot_eig, rot_eig)
    proj_scr = np.zeros((natm, 3, 6))      #Representing the D matrix in Gaussian
    proj_scr[:, (0, 1, 2), (0, 1, 2)] = 1        #Representative Uniform Projection
    proj_scr[:, :, 3] = (rot_coord[:, 1, :, 2] - rot_coord[:, 2, :, 1])   #Representative Rotating Projection
    proj_scr[:, :, 4] = (rot_coord[:, 2, :, 0] - rot_coord[:, 0, :, 2])
    proj_scr[:, :, 5] = (rot_coord[:, 0, :, 1] - rot_coord[:, 1, :, 0])
    proj_scr *= np.sqrt(mol_weight)[:, None, None]
    proj_scr.shape = (-1, 6)
    proj_norm = np.linalg.norm(proj_scr, axis=0) #(6,) Normalize by column
    proj_norm[proj_norm < 1e-15] = 1.0
    proj_scr /= proj_norm  #D Matrix(3N,6),Column-wise normalization, no unit,to these new internal coordinates(only Translation + rotation)

    e_tr, _ = np.linalg.eigh(proj_scr.T @ theta @ proj_scr)     # to these new internal coordinates(only Translation + rotation)
    no_freq = np.sqrt(np.abs(e_tr * E_h * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e_tr > 0) * 2 - 1)
    #no_freq    # Translational and rotational frequencies given by diagonalization
    
    D = proj_scr.copy()   # shape (3N, 6)
    Qfull, _ = np.linalg.qr(D, mode='complete')
    proj_inv = Qfull[:, 6:]      # (3N, 3N-6) D Matrix(3N,3N-6), no unit, to these new internal coordinates(remove Translation + rotation)
    
    e, q = np.linalg.eigh(proj_inv.T @ theta @ proj_inv)   #((3N-6,3N) @ (3N,3N) @ (3N,3N-6)) to these new internal coordinates(remove Translation + rotation)
   
    real_freq_cm_1 = np.sqrt(np.abs(e * E_h * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e > 0) * 2 - 1)
    #real_freq_cm_1               #Only vibration frequency

    q_unnormed = np.einsum("AtQ, A -> AtQ", (proj_inv @ q).reshape(natm, 3, (proj_inv @ q).shape[-1]), 1 / np.sqrt(mol_weight))   
    q_unnormed = q_unnormed.reshape(-1, q_unnormed.shape[-1]) #(3N,3N-6), unit amu^(-1/2), it is l_CART, which are the normal modes in cartesian coordinates
    
    norm_q = np.linalg.norm(q_unnormed, axis=0)   #(3N-6,)
    #Minimums are not normalized, they remain unchanged.
    norm_q[norm_q < 1e-15] = 1.0       #Prevent division by zero (linear mol, mode: 3N-5)
    q_normed = q_unnormed / norm_q        #Normalized normal coordinates
    #q_normed     # Normalized normal coordinates
    
    FactIR = (1/0.20819434)**2 * 42.255 # 1 (Debye/Angstrom)^2 amu^-1 = 42.255 KM/mol, there use e*Angstrom, 1 e*Angstrom^2 =(1/0.20819434)**2 Debye^2 
    eigen_L = (proj_inv @ q).reshape(natm, 3, (proj_inv @ q).shape[-1]) #(N,3,3N-6),D Matrix(3N,3N-6) @ (3N-6,3N-6)
    
    return real_freq_cm_1,q_unnormed,eigen_L

def XYZ_output(filename, line_2, coordinates, attype, scaled_Freq, ir_intensities, scale_factors):
    # Function: Output XYZ format file (with spectral data)
    # Description: Write optimized geometry, charges, frequencies, and IR intensities
    #              into a .xyz file

    fw = open(filename, 'w')
    fw.write(str(len(attype)) + '\n')
    fw.write(line_2 + '\n')
    for i in range(len(attype)):
        fw.write(str(attype[i]) + ' ' + ' '.join('%.6f' % c for c in coordinates[i]) + ' ' + '\n')
    fw.write('\n')
    fw.write('scaled_Freq(cm^-1) Inten(KM/mol) scale_factor \n')
    for j in range(len(scaled_Freq)):
        fw.write(f'{scaled_Freq[j]:.4f} {ir_intensities[j]:.4f} {scale_factors[j]:.3f} \n')
    fw.close()
    
def read_coord(file):
    # Function: Read XYZ file
    # Description: Parse .xyz file, return atomic coordinates, atomic types, and comment line

    str_type = []
    position = []
    fp = open(file, 'r')
    lines = fp.readlines()
    
    natom = int(lines[0].split()[0])
    second_line = lines[1].strip()
    match = re.search(r'Charge\s*=\s*([+-]?\d+)', second_line)
    if match:
        charge = int(match.group(1))
    else:
        raise ValueError(f"{file}: Charge is not found in the second line. Use Charge=0, Charge=1, or Charge=-1.")

    for line in lines[2:2+natom]:
        line = line.strip().split()
        at = line[0]
        str_type.append(at)
        position.append(list(map(float, line[1:4])))
   
    fp.close()
    return position, str_type, second_line, charge

def chem_formula(str_type):
    # Function: Generate chemical formula
    # Description: Generate molecular formula string from atom type list (e.g., C10H8)

    symbol = copy.deepcopy(str_type)        
    formula = ''
    if symbol.count('C') != 0:
        formula += 'C' + str(symbol.count('C'))
    if symbol.count('H') != 0:
        formula += 'H' + str(symbol.count('H'))
    if symbol.count('N') != 0:
        formula += 'N' + str(symbol.count('N'))
    if symbol.count('O') != 0:
        formula += 'O' + str(symbol.count('O'))
    if symbol.count('Si') != 0:
        formula += 'Si' + str(symbol.count('Si'))
    if symbol.count('Mg') != 0:
        formula += 'Mg' + str(symbol.count('Mg'))
    if symbol.count('Fe') != 0:
        formula += 'Fe' + str(symbol.count('Fe'))
        
    if len(formula) >= 2:
      if not formula[-2].isdigit() and int(formula[-1]) == 1 :
        formula = formula[:-1]
    
    return formula

def num2symbol(num_type, charge):
    # Function: Convert atomic numbers to symbols and molar masses
    # Description: Convert atomic numbers (e.g., 6 -> C, 1 -> H) to symbols with molar masses

    atom_map = {
        6: ("C", 12.011),
        1: ("H", 1.008),
        8: ("O", 15.999),
        7: ("N", 14.007),
    }

    symbol_type = []
    mol_weight = []

    for at in num_type:
        symbol, weight = atom_map.get(int(at), ("X", 0.0))  # Unknown element default "X", mass 0.0
        # Add charge label based on charge state (e.g., C+, C-)
        if charge == 0:
            symbol_type.append(symbol)
        elif charge > 0:
            symbol_type.append(f"{symbol}{charge}+")
        elif charge < 0:
            symbol_type.append(f"{symbol}{abs(charge)}-")

        mol_weight.append(weight)
    return symbol_type, mol_weight
        
def first_deriva_energy(fun):
    # Function: First derivative (gradient)
    # Description: Return the first derivative of the energy function (atomic forces), 
    #              with optional JIT acceleration

    if jit_flag:
        return jax.jit(jax.grad(fun)) 
    else:
        jax.config.update('jax_disable_jit', True)   # Disable JIT acceleration
        return jax.grad(fun)
        
def second_derivative(fun):
    # Function: Second derivative (Hessian)
    # Description: Return the second derivative of the energy function (Hessian matrix),
    #              with optional JIT acceleration

    if jit_flag:
        return jax.jit(jax.hessian(fun))
    else:
        jax.config.update('jax_disable_jit', True)   # Disable JIT acceleration
        return jax.hessian(fun)   


if __name__ == "__main__":
          
    # Initialize the argument parser for handling command-line inputs
    parser = argparse.ArgumentParser(
        description="Parameters for calc harm IR."
        )
        
    parser.add_argument(
        "-d",
        "--delta_p",
        help="dipole delta (finite difference step for dipole derivatives)",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "-s",
        "--save_path",
        help="path of save IR result",
        type=str,
        default='../outputs/IR',
    )

    parser.add_argument(
        "-n",
        "--xyz_file",
        help="path of the xyz file that needs to be calculated",
        type=str,
        default='../inputs/XYZ',
    )

    parser.add_argument(
        "-e",
        "--model_e",
        help="energy model (NNFF model path)",
        type=str,
        default='../model',
    )

    parser.add_argument(
        "-p",
        "--model_p",
        help="dipole model (EPNN model path)",
        type=str,
        default='../model',
    )
        
    parser.add_argument(
            "-fm",
            "--fmax",
            help="Optimized parameters (geometry optimization convergence threshold)",
            type=float,
            default=0.0005,
        )
        
    args = parser.parse_args()
    jit_flag = True   # use JIT acceleration
    opt_flag = True          # perform geometry optimization

    # Get list of all .xyz files in the folder
    for root, dirs, files in os.walk(args.xyz_file):
      for file in tqdm(files, desc=f"Computing IR spectra", unit="file"):
        if file.endswith('.xyz'):
            filename_i = os.path.join(root, file)    
            # Process each file with a progress bar
    
            try:        
                time_start = time.time()
                # Read input file and check element types                
                position_i, str_type_i, second_line_i, charge_input = read_coord(filename_i)
                key_i = os.path.basename(filename_i).split('.xyz')[0]
                position_i = jnp.array(position_i)
                formula0 = chem_formula(str_type_i)
                
                # Check if the molecule contains any element other than C and H
                element_set = set(str_type_i)
                allowed_elements = {'C', 'H'}

                if not element_set.issubset(allowed_elements):
                    print(f"{key_i}: contains elements other than C and H ({element_set - allowed_elements}), skipping")
                    continue  # Skip this molecule and move to the next

                # Optional: Check if the molecule contains at least one carbon and one hydrogen
                # (PAHs must have both C and H)
                if 'C' not in element_set or 'H' not in element_set:
                    print(f"{key_i}: not a pure hydrocarbon (missing C or H), skipping")
                    continue
                    
                charge_0 = charge_input   # Molecular charge state
                # Validate charge state input: only -1, 0, and +1 are supported; exit with error otherwise
                if int(charge_0) not in [-1, 0, 1]:
                    print(f"Error: Unsupported Charge state '{charge_0}'. Only -1, 0, and +1 are supported.")
                    print("Usage: put Charge=0, Charge=1, or Charge=-1 in the second line of each .xyz file.")
                    sys.exit(1)
                
                if int(charge_0) == 0:
                    model_e_file = os.path.join(args.model_e, f"model_energy_neutral_0.pkl")   # NNFF energy model 
                    model_p_file = os.path.join(args.model_p, f"model_dipole_neutral_0.pkl")   # EPNN dipole model
                elif int(charge_0) == 1:
                    model_e_file = os.path.join(args.model_e, f"model_energy_cation_+1.pkl")   # NNFF energy model 
                    model_p_file = os.path.join(args.model_p, f"model_dipole_cation_+1.pkl")   # EPNN dipole model
                elif int(charge_0) == -1:
                    model_e_file = os.path.join(args.model_e, f"model_energy_anion_-1.pkl")   # NNFF energy model 
                    model_p_file = os.path.join(args.model_p, f"model_dipole_anion_-1.pkl")   # EPNN dipole model
                
                PICKLE_energy = model_e_file   
                model_info_e = pickle.load(open(PICKLE_energy, "rb"))

                N_ENSEMBLE = 5       # Number of NNFF ensemble models
                max_neighbors = 25   # Maximum number of neighbors

                PICKLE_dipole = model_p_file  
                model_info_dipole = pickle.load(open(PICKLE_dipole, "rb"))
                N_MAX_D = 6
                N_BATCH_D = 1
                
                # Get element mapping tables (energy and dipole models may differ)
                sorted_elements_e = model_info_e.sorted_elements
                symbol_map_e = {s: i for i, s in enumerate(sorted_elements_e)} 
                if hasattr(model_info_dipole, "sorted_elements"):
                    sorted_elements_p = model_info_dipole.sorted_elements
                else:
                    sorted_elements_p = sorted_elements_e   # Fallback: use energy model's element list
                symbol_map_p = {s: i for i, s in enumerate(sorted_elements_p)}
                    
                # Create the object that will generate descriptors for each configuration.
                descriptor_generator = PowerSpectrumGenerator(
                    model_info_e.n_max, model_info_e.r_cut, len(model_info_e.sorted_elements), max_neighbors
                )    
                   
                # Create the NNFF model. The number and types of the parameters is completely
                #     dependent on the kind of model used.
                # Build NNFF core model: ResNet architecture
                core_model = ResNetCore(model_info_e.core_widths)

                individual_model = NeuralILwithMorse(
                            len(model_info_e.sorted_elements),
                            model_info_e.embed_d,
                            model_info_e.r_cut,
                            descriptor_generator,
                            descriptor_generator.process_some_data,
                            core_model,
                            )
                dynamics_model = PlainEnsemblewithMorse(individual_model, N_ENSEMBLE)
                model_params=model_info_e.params

                # Initialize EPNN model (dipole moment prediction)
                # Initialize the bias of the linear layer to something vaguely reasonable to
                #    avoid wasting a few dozen epochs just centering the predicted energies.
                @jax.jit
                def individual_energy_calculator(params, positions, types, cell):
                    return individual_model.apply(
                        params,
                        positions,
                        types,
                        cell,
                        method=individual_model.calc_potential_energy,
                    )
                
                def get_n_models(model_params):
                    """Return the number of models contained in a plain ensemble.

                    Args:
                        model_params: The tree of parameters of the full ensemble including
                            all models.

                    Returns:
                        An integer with the number of models contained in model_params.
                    """
                    return model_params["params"]["neuralil"]["denormalizer"]["bias"].shape[0]
                
                def unpack_params(model_params):
                    "Extract a list of individual parameters sets from the ensemble parameters."
                    n_models = get_n_models(model_params)
                    nruter = []
                    for i_model in range(n_models):
                        subparams = jax.tree_map(operator.itemgetter(i_model), model_params)
                        individual_params = dict(params=subparams["params"]["neuralil"])
                        nruter.append(flax.core.freeze(individual_params))
                    return nruter
                
                individual_params = unpack_params(model_params)
                
                n_types_p = len(sorted_elements_p)
                descriptor_generator_dipole = PowerSpectrumGenerator(N_MAX_D, model_info_dipole.r_cut, n_types_p, max_neighbors).process_data

                @jax.jit
                def get_bessel_descriptor(pos,t,c):
                    descriptor=descriptor_generator_dipole(pos, t, c)
                    return descriptor

                # Build EPNN model: MLP updater + message passing layers + charge passing layers
                update_model_dipole = MLP(model_info_dipole.update_model_widths)

                message_generation_models_diople = tuple([
                        ResNetCore(model_info_dipole.message_generator_widths)
                            for _ in range(model_info_dipole.n_passes)
                            ])

                electron_pass_generation_models_diople = tuple([
                        ResNetCore(model_info_dipole.pass_generator_widths)
                            for _ in range(model_info_dipole.e_passes)
                            ])

                model_dipole = EPNN(
                            n_types_p,
                            model_info_dipole.embed_dim,
                            update_model_dipole,
                            message_generation_models_diople,
                            electron_pass_generation_models_diople,
                                            )
               
                delta_p = args.delta_p   # Finite difference step for dipole derivatives (Angstrom) 
                save_path = args.save_path     # Output directory
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
    
                # Print start message
                opt_xyz_file = os.path.join(save_path, f'{key_i}.xyz')
                if os.path.exists(opt_xyz_file):
                    print(f"{opt_xyz_file} already exists, skipping")
                    continue

                print(f'Running: {filename_i}')

                # Geometry optimization (using ASE + BFGS algorithm)
                if opt_flag:
                    # Initialize the ASE (Atomic Simulation Environment) calculator
                    calculator = NeuralILASECalculator(dynamics_model, model_info_e, max_neighbors, charge_0, n_devices=1)
                    # Attach the calculator to the atomic structure object
                    atoms = read(filename_i)
                    atoms.set_calculator(calculator)
                    logfile_i = os.path.join(save_path, f'{key_i}.log')
                    opt = BFGS(atoms, logfile=logfile_i)
                    
                    # Track the structure with the smallest Fmax (for fallback if optimization fails)
                    state = {
                            "best_fmax_pos": atoms.get_positions().copy(),
                            "nan_flag": False,
                            "best_fmax": np.inf,
                            "fail_msg": ""
                        }

                    def check_finite():
                        # Check if atomic coordinates, energy, and forces are finite; record structure with minimum Fmax
                        pos = atoms.get_positions()

                        # Check if coordinates are finite
                        if not np.all(np.isfinite(pos)):
                            state["nan_flag"] = True
                            state["fail_msg"] = "positions contain nan/inf"
                            raise RuntimeError(state["fail_msg"])
                        
                        e = atoms.get_potential_energy()
                        f = atoms.get_forces()
               
                        ok = np.isfinite(e) and np.all(np.isfinite(f))
                        if ok:
                            fmax_now = np.sqrt((f ** 2).sum(axis=1)).max()

                            # Record the coordinates corresponding to the minimum value of fmax
                            if fmax_now < state["best_fmax"]:
                                state["best_fmax"] = fmax_now
                                state["best_fmax_pos"] = pos.copy()
                        else:
                            state["nan_flag"] = True
                            state["fail_msg"] = f"calculator results not finite: energy={e}"
                            raise RuntimeError(state["fail_msg"])

                    try:
                        # First, check it once to avoid not having the historical optimal structure
                        check_finite()

                        # Check once every 1 step
                        opt.attach(check_finite, interval=1)
                        converged = opt.run(fmax=args.fmax, steps=1000)
                    except Exception as err:
                        converged = False
                        state["nan_flag"] = True
                        if not state["fail_msg"]:
                            state["fail_msg"] = str(err)

                    # Conduct a final check again and update the variable "best_fmax"
                    final_ok = True
                    try:
                        pos_final = atoms.get_positions()
                        e_final = atoms.get_potential_energy()
                        f_final = atoms.get_forces()

                        if not np.all(np.isfinite(pos_final)):
                            final_ok = False
                        if not np.isfinite(e_final):
                            final_ok = False
                        if not np.all(np.isfinite(f_final)):
                            final_ok = False

                        if final_ok:
                            fmax_final = np.sqrt((f_final ** 2).sum(axis=1)).max()
                            if fmax_final < state["best_fmax"]:
                                state["best_fmax"] = fmax_final
                                state["best_fmax_pos"] = pos_final.copy()

                    except Exception as err:
                        final_ok = False
                        if not state["fail_msg"]:
                            state["fail_msg"] = str(err)

                    # If optimization fails, fall back to the structure with minimum Fmax
                    if (not converged) or (not final_ok) or state["nan_flag"]:
                        atoms.set_positions(state["best_fmax_pos"])
                        position0 = state["best_fmax_pos"].copy()
                    else:
                        position0 = atoms.get_positions().copy()

                    str_type = atoms.get_chemical_symbols().copy()

                    # Delete log entries
                    if os.path.exists(logfile_i):
                        os.remove(logfile_i)
                else:
                    str_type = str_type_i
                    position0 = position_i
                    
                # Prepare data for computation (atom type mapping, masses, charge initialization, etc.)
                cell0  = jnp.zeros((3,3),dtype=jnp.float32)
                natom = len(str_type)
                
                # Build atomic mass and atomic number lists from chemical symbols
                mol_weight0 = []
                num_type = []
                for at in str_type:               
                    if str(at)== 'C':
                        mol_weight0.append(12.011)
                        num_type.append(6)
                    elif str(at)== 'H':
                        mol_weight0.append(1.008)
                        num_type.append(1)
                    elif str(at) == 'O':
                        mol_weight0.append(15.999)
                        num_type.append(8)
                    elif str(at) == 'N':
                        mol_weight0.append(14.007)
                        num_type.append(7)
                    elif str(at) == 'Si':
                        mol_weight0.append(28.085)
                        num_type.append(14)
                    elif str(at) == 'Mg':
                        mol_weight0.append(24.305)
                        num_type.append(12)
                    elif str(at) == 'Fe':
                        mol_weight0.append(55.845)
                        num_type.append(26)
                
                num_type = jnp.array(num_type) 
                symbol_type, _ = num2symbol(num_type, charge_0)     # The symbol_type is C, C+, or C-, while the str_type is C, but there is no C+.
                type_train0_e = jnp.array([symbol_map_e[s] for s in symbol_type])
                type_train0_p = jnp.array([symbol_map_p[s] for s in symbol_type])
                init_charges0 = jnp.full(natom, charge_0/natom)
                mol_weight0 = jnp.array(mol_weight0)
                to_Debye =  4.80320454    #e*Angstrom is converted to Debye
                
                def count_enery(X) :
                    # Predict potential energy using the ensemble average of NNFF models.
                    position = X
                    energies_pre = []
                    for p in individual_params:

                        energy_pre1 = individual_energy_calculator(
                        p, position, type_train0_e, cell0
                        )   
                        energies_pre.append(energy_pre1)

                    energies_pre = jnp.array(energies_pre)
                    energy_avg  = jnp.sum(energies_pre, axis=0) / energies_pre.shape[0]
                    return energy_avg
                
                def Grad(delta,position_0):
                    # Generate displaced configurations for finite difference dipole derivatives.
                    position_list = []
                    cell_list = []
                    type_train_list = []
                    init_charges_list = []
                    for i in range(0,position_0.shape[0]):
                        for j in range(0, position_0.shape[1]):
                            position2 = np.array(position_0)
                            position1 = np.array(position_0)
                            position2[i,j] += delta
                            position1[i,j] -= delta
                            
                            center_mass2 = jnp.sum(position2.T * mol_weight0, axis=1) / jnp.sum(mol_weight0)
                            position2 = position2 - center_mass2
                            center_mass1 = jnp.sum(position1.T * mol_weight0, axis=1) / jnp.sum(mol_weight0)
                            position1 = position1 - center_mass1
                          
                            position_list.extend([position1,position2])
                            cell_list.extend([cell0,cell0])
                            type_train_list.extend([type_train0_p,type_train0_p])
                            init_charges_list.extend([init_charges0,init_charges0])
                        
                    return position_list,cell_list,type_train_list,init_charges_list
                    
                def first_derivadipole(delta,position_0):
                    # Compute the first derivative of the dipole moment with respect to Cartesian coordinates.      
                    position_list,cell_list,type_train_list,init_charges_list = Grad(delta,position_0) 
                    positions = jnp.array(position_list)
                    cells = jnp.array(cell_list)
                    type_trains = jnp.array(type_train_list)
                    init_charges = jnp.array(init_charges_list)
                    
                    descriptors=[]
                    for i in range(0,positions.shape[0]):
                        descriptor = get_bessel_descriptor(positions[i], type_trains[i], cells[i])
                        descriptors.append(descriptor)

                    # Descriptor tensors are reshaped to flatten to bessel descriptors
                    descriptors = jnp.asarray(descriptors)          
                    descriptors = descriptors.reshape(*descriptors.shape[:2],-1)
                    
                    test_graphs = create_batched_graph_list(
                    positions,
                    cells,
                    descriptors,
                    type_trains,
                    init_charges,
                    model_info_dipole.r_cut,
                    model_info_dipole.r_switch,
                    model_info_dipole.node_state_dim,
                    N_BATCH_D,
                    )

                    pred_charges = jnp.asarray(jnp.squeeze(jnp.vstack(list(map(
                        lambda graph: get_nodes_from_graph(model_dipole.apply(model_info_dipole.params, graph))[0],
                    test_graphs
                        )))))

                    pred_charges = jnp.asarray(pred_charges) # calculated chagres
                    pred_charges = jnp.expand_dims(pred_charges,axis=-1)

                    dipole_monment_pred=[]
                    for i in range(pred_charges.shape[0]):
                        dipole_monment_pred.append(sum(pred_charges[i]*positions[i]))
                        
                    dipole_monment_pred=jnp.array(dipole_monment_pred)        
                    dipole_monment_pred = dipole_monment_pred.reshape(natom*3,2,3)        
                    
                    dipolederiv = []
                    for i in range(0,natom*3):
                        one_atom_deriv = []
                        for j in range(0,3):
                            one_atom_deriv.append( (dipole_monment_pred[i,1,j] - dipole_monment_pred[i,0,j]) / (2*delta) )
                        dipolederiv.append(one_atom_deriv)
                    dipolederiv = np.array(dipolederiv)
                    dipolederiv = dipolederiv.reshape(natom*3,3)
                    
                    del descriptors,test_graphs
                    return dipolederiv    #(3N,3) e*Angstrom/Angstrom
                
                # Compute Hessian matrix via automatic differentiation (3N, 3N) 
                second_f = second_derivative(count_enery)
                second_deriva_result_e_xyz = np.array(second_f(position0))   #(N,3,N,3) unit ev/Angstrom^2
                second_deriva_result_e_xyz = second_deriva_result_e_xyz.reshape(natom*3,natom*3)  #(3N,3N) unit ev/Angstrom^2
                # Convert Hessian to harmonic frequencies and normal modes
                Harm_freq,eigen_q,eigen_L = Hess2freq(second_deriva_result_e_xyz,position0,mol_weight0)    #eigen_q,(3N,3N-6), amu^-1/2
                del second_f,second_deriva_result_e_xyz
                gc.collect()
                
                # Compute dipole moment and derivatives
                first_deriva_result_p_xyz = np.array(first_derivadipole(delta_p,position0))  #(3N,3) e*Angstrom/Angstrom 
                first_deriva_result_normal_p_xyz = np.einsum("Pi, PQ -> Qi", first_deriva_result_p_xyz, eigen_q) #(3N-6,3) e*Angstrom Angstrom^-1 amu^-1/2
                
                # Compute IR intensities
                ir_intensities = (first_deriva_result_normal_p_xyz ** 2).sum(axis=1) #(3N-6,) e*Angstrom^2 Angstrom^-2 amu^-1
                ir_intensities = (ir_intensities * to_Debye**2) * 42.255   #(3N-6,)  FactIR  1 Debye^2  Angstrom^-2 amu^-1 = 42.255 KM mol^-1

                # Write output file
                ft_CH = 0.960  #Bauschlicher_2018 (2500-)   
                ft_CC = 0.952  #Bauschlicher_2018 (1111.11-2500)    
                ft_CHoop = 0.956 #Bauschlicher_2018 (0-1111.11)    

                scaled_Freq = np.where(Harm_freq > 2500, Harm_freq * ft_CH,
                                               np.where(Harm_freq > 1111.11, Harm_freq * ft_CC, Harm_freq * ft_CHoop))
                                               
                scale_factors = np.where(Harm_freq > 2500, ft_CH,
                                 np.where(Harm_freq > 1111.11, ft_CC, ft_CHoop)) 
                                 
                line_2 = f'{second_line_i}'
                XYZ_output(opt_xyz_file, line_2, position0, str_type, scaled_Freq, ir_intensities, scale_factors)
                
                # Print timing information
                time_end = time.time() 
                time_diff = time_end - time_start
                hours = time_diff // 3600
                minutes = (time_diff % 3600) // 60
                seconds = time_diff % 60 
                print(f'Finish: {filename_i}')
                print("Time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")  

            except Exception as err:
                print(f"{filename_i} failed: {err}")
                continue    