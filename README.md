# Project_Fullerenes

This repository contains all input files needed to reproduce the nonadiabatic molecular dynamics (NA-MD) simulations used to study nonradiative excited-state relaxation in the fullerene series C20, C60, C70, C76, C84, C86, and C90.


The associated study investigates how dense manifolds of electronic excited states and coherent population spreading govern energy relaxation in fullerenes. Using surface-hopping methodologies within TD-DFT framework, we show that electronic Shannon entropy provides a mechanistic explanation of size-dependent relaxation trends across the series.


We recommend checking out this tutorial for detailed instructions on how to run each step of the workflow.[Libra CompChemCyberTraining](https://github.com/compchem-cybertraining/Tutorials_Libra/tree/master/6_dynamics/2_nbra_workflows). Detailed explanations about installation and running the CP2K inputs can be found in [here](https://github.com/compchem-cybertraining/Tutorials_CP2K).


## 1. Molecular dynamics

`1_MD`contains all CP2K inputs used to generate the ab initio molecular dynamics (AIMD) trajectories for each fullerene.
Each fullerene has its own compressed folder:

`1_c20.tar.bz2` 
`2_c60.tar.bz2`  
`3_c70.tar.bz2`  
`4_c76.tar.bz2`  
`5_c84.tar.bz2`  
`6_c86.tar.bz2`  
`7_c90.tar.bz2`

to unpack the data, use:

    tar -xf 1_c20.tar.bz2

Inside each folder you will find:

`c20.xyz` is geometry file: the initial structure used to start the AIMD simulation.

`md.inp` is CP2K input file: defines all simulation parameters.

`submit.slm` is SLURM script: example job script for running the MD calculation on HPC.

`C20-pos-1.xyz` is trajectory file: output file containing ~6000 MD snapshots.

`align_md_trajectory.py` is alignment script: removes overall translations and rotations to produce an aligned trajectory required for computing time-overlaps and NACs.

The aligned trajectory ensures consistent geometrical reference frames across all time steps, which is essential for accurate overlap calculations in later steps.

For this step, unpack the contents of the trajectory file, and execute the following command:
    
    sbatch submit.slm
    

## 2. Molecular orbitals overlaps 

`2_electronic_structure_calculations` contains all inputs needed to compute molecular orbital (MO) overlaps and time-overlaps along the aligned AIMD trajectory. These calculations are required for generating the excited-state information used in the NAC and NA-MD steps.

to unpack the data, use:

    tar -xf 1_c20.tar.bz2

For c20 fullerene, the folder includes:

`C20-md-aligned.xyz` is aligned AIMD trajectory: This is the trajectory obtained after removing translation and rotation during Step 1.

`cp2k_temp.inp` is CP2K input file: it's used to compute molecular orbitals at each snapshot.

`distribute_jobs.py` is Python job distributor : Script that you need to modify by specifying the initial step, final step, and number of jobs. Based on these settings, Libra automatically splits the trajectory and prepares separate folders for each job.

`run_template.py` is run template: it's used to perform MO overlap calculations. The script will be copied and filled with the correct step indices for each job.

`submit_template.slm` is SLURM submission script: Submission script that runs run.py inside each job folder. The `run.py` file is automatically generated from `run_template.py` and populated with the required parameters.

To run the MO overlap and TD-DFT calculations, simply execute:

    python distribute_jobs.py


## 3. Nonadiabatic couplings

`3_nonadiabatic_coupling_calculation` contains all inputs needed to compute nonadiabatic couplings (NACs) between excited states along the aligned AIMD trajectory. These NACs are essential for the subsequent NA-MD simulations.

to unpack the data, use:

    tar -xf 1_c20.tar.bz2

For c20 fullerene, the folder includes:

`C20.py` is Python script: Computes the NACs using the molecular orbital overlaps generated in Step 2.
Make sure the paths to the Step 2 outputs are correctly specified inside this script.

`submit_template.slm` is SLURM submission script: Example job script to run the NAC calculation on an HPC cluster.

Submit the job using:

    sbatch submit_template.slm

## 4. Nonadiabatic molecular dynamics

`4_nonadiabatic_dynamics_calculations` contains all inputs needed to run nonadiabatic molecular dynamics (NA-MD) simulations for the fullerenes. This step updates the electronic populations along the nuclear trajectories using trajectory surface hopping (TSH) methods.

To unpack the data for C20, use:
 
      tar -xf 1_c20.tar.bz2
      
Folder contents (C20 example): The folder includes subfolders for each TSH method:

`fssh`, `ida`, `msdm`, `fssh2`, `gfsh`, `msdm_gfsh`

Inside each method folder, you will find:

`step4_*.py`: Python script to start the NA-MD for a single trajectory using the corresponding TSH scheme. Make sure the paths to the NAC outputs from Step 3 are correctly set in this script.

`submit_template.slm`: Example SLURM job script to run the NA-MD on an HPC cluster.

`recipes` and `recipes_latest`: Contains predefined recipes for the different TSH schemes. Do not modify this folder; it must remain in the same directory as the NA-MD runs. 
   
Submit the NA-MD job using:

    sbatch submit.slm

## 5. Visualizing

The Python files in folder `5_analysis` are multiple files that are used for analyzing and plotting the properties of the generated data from previous steps. 

`PD_NACs_Egaps_visualization.py`: This Python script is used to analyze and visualize the plots for probability distribution (PD) of NACs between  all adjacent (i, i+1) states excluding S0; and energy gaps between all adjacent (i, i+1) states excluding S0.

`Average_Excess_Energy.ipynb`: This code analyzes the time-dependent excess electronic energy of the fullerene for multiple nonadiabatic dynamics methods and the BLLZ. It computes population-weighted energies, fits their relaxation using single and double-exponential fitting, and visualizes both the raw data and fitted curves to extract relaxation times.

`Shannon_entropy.ipynb`: Calculates and visualizes the time evolution of the normalized Shannon entropy using population data from multiple nonadiabatic dynamics methods. It averages the entropy over multiple trajectories for each method to quantify how electronic state populations spread and disorder increases over time.

`TRPES.ipynb`: This script computes and visualizes the Time-Resolved Photoelectron Spectra (TRPES) using multiple nonadiabatic dynamics methods, based on trajectory population data from NAMD simulations. It also calculates the average excitation energy over time and overlays it on the TRPES plots to analyze energy evolution across different methods.

`spectra_dynamics.py`: This code is designed to analyze and visualize the dynamics and spectral properties of molecular systems. Specifically, it loads adiabatic energies from simulations, calculates excess energy evolution over time for multiple nonadiabatic dynamics methods, computes the density of states (DOS) using vibrational Hamiltonians, and processes UV-VIS spectra from CP2K log files, combining all these results into comprehensive plots that show energy dynamics, DOS, and spectral features together.
