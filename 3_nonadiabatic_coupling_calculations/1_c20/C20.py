import os, glob, time, h5py, warnings
import numpy as np
import scipy.sparse as sp
from libra_py import units, data_stat, influence_spectrum, data_conv
import matplotlib.pyplot as plt
from liblibra_core import *
from libra_py.workflows.nbra import step3
import libra_py.packages.cp2k.methods as CP2K_methods
#from IPython.display import clear_output
import multiprocessing as mp

import util.libutil as comn
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.workflows.nbra.decoherence_times as decoherence_times



params_active_space = {
    'lowest_orbital': 17, 'highest_orbital': 54, 'num_occ_orbitals': 11, 'num_unocc_orbitals': 10,
    'path_to_npz_files': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/c20-MOs60/step2/res', 
    'logfile_directory': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/c20-MOs60/step2/all_logfiles',
    'path_to_save_npz_files': os.getcwd()+'/new_res_c20'
}
new_lowest_orbital, new_highest_orbital = step3.limit_active_space(params_active_space)


params_mb_sd = {
          'lowest_orbital': new_lowest_orbital, 'highest_orbital': new_highest_orbital, 
          'num_occ_states': 6, 'num_unocc_states': 5,
          'isUKS': 0, 'number_of_states': 40, 'tolerance': 0.0, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': 4,
          'is_many_body': True, 'time_step': 0.5, 'es_software': 'cp2k',
          'path_to_npz_files': params_active_space['path_to_save_npz_files'],
          'logfile_directory':'/projects/academic/alexeyak/kosar/cp2k/fullerenes/c20-MOs60/step2/all_logfiles',
          'path_to_save_sd_Hvibs': os.getcwd()+'/res-mb-sd-c20',
          'outdir': os.getcwd()+'/res-mb-sd-c20', 'start_time': 1000, 'finish_time': 4998, 'sorting_type': 'energy',
         }

step3.run_step3_sd_nacs_libint(params_mb_sd)
