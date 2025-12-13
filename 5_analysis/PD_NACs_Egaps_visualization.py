#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.sparse as sp
from liblibra_core import MATRIX
from libra_py import units, data_stat
import libra_py.packages.cp2k.methods as CP2K_methods

paths = {
    'C$_{20}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c20/res-mb-sd-c20/',
    'C$_{60}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c60/res-mb-sd-c60/',
    'C$_{70}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c70/res-mb-sd-c70/',
    'C$_{76}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c76/res-mb-sd-c76/',
    'C$_{84}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/res-mb-sd-c84/',
    'C$_{86}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c86/res-mb-sd-c86/',
    'C$_{90}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c90/res-mb-sd-c90/'
}

cmap = cm.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=0, vmax=len(paths)-1)
def get_color(i): return cmap(norm(i))

def detect_index_range(folder):
    files = glob.glob(os.path.join(folder, "Hvib_ci_*_re.npz"))
    idxs = []
    for f in files:
        m = re.search(r'Hvib_ci_(\d+)_re\.npz', os.path.basename(f))
        if m:
            idxs.append(int(m.group(1)))
    return (min(idxs), max(idxs)) if idxs else (None, None)

def plot_adjacent_gaps(paths_dict, out_png='energy_gap_adjacent_excited_fullerenes_coolwarm.png'):
    plt.figure(figsize=(6,4))

    custom_colors = {"C$_{76}$": "green"}

    for i, (label, folder) in enumerate(paths_dict.items()):

        color = custom_colors.get(label, get_color(i))

        istep, fstep = detect_index_range(folder)
        if istep is None:
            print(f"[Skip] {label}: No Hvib_ci_* files found.")
            continue

        params = {
            "path_to_energy_files": folder, "dt": 1.0,"prefix": "Hvib_ci_", "suffix": "_re", "istep": istep, "fstep": fstep}

        try:
            t, E = CP2K_methods.extract_energies_sparse(params)
        except Exception as e:
            print(f"[Error] {label}: {e}")
            continue

        E_eV = E * units.au2ev
        nst = E_eV.shape[1]

        if nst < 3:
            print(f"[Skip] {label}: not enough excited states")
            continue

        mats = []
        for ist in range(1, nst-1):   # S1-S2, S2-S3, ...
            gaps = np.abs(E_eV[:, ist+1] - E_eV[:, ist])
            for val in gaps:
                x = MATRIX(1,1)
                x.set(0,0,float(val))
                mats.append(x)

        # Normalize (A=1)
        bin_supp, dens, cum = data_stat.cmat_distrib(
            mats, 0, 0, 0, 0,
            60,       # number of bins
            0.005     # bin width
        )

        dens = np.array(dens)
        bin_width = 0.005
        area = np.sum(dens * bin_width)
        if area > 0:
            dens = dens / area

        plt.plot(bin_supp, dens, label=label, color=color, linewidth=2.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 1e1)
    plt.xlabel('Energy gap (S$_i$, S$_{i+1}$) (eV)', fontsize=24)
    plt.ylabel('PD (1/eV)', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.show()

     
plot_adjacent_gaps(paths)   


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import glob, os
from matplotlib.ticker import LogLocator, NullFormatter
from liblibra_core import MATRIX
from libra_py import units, data_stat
import matplotlib.cm as cm
import matplotlib.colors as mcolors

paths = {
    'C$_{20}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c20/res-mb-sd-c20/',
    'C$_{60}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c60/res-mb-sd-c60/',
    'C$_{70}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c70/res-mb-sd-c70/',
    'C$_{76}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c76/res-mb-sd-c76/',
    'C$_{84}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c84/res-mb-sd-c84/',
    'C$_{86}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c86/res-mb-sd-c86/',
    'C$_{90}$': '/projects/academic/alexeyak/kosar/cp2k/fullerenes/step3_data_9_10_2025/c90/res-mb-sd-c90/'
}

cmap = cm.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=0, vmax=len(paths)-1)

def get_color(i):
    return cmap(norm(i))

custom_colors = {"C$_{76}$": "green"}

plt.figure(figsize=(6, 4))

for idx, (system, base_dir) in enumerate(paths.items()):
    nac = []
    
    for nac_file in glob.glob(os.path.join(base_dir, 'Hvib_ci*im*')):
        hvib = sp.load_npz(nac_file).todense().real
        nst = hvib.shape[0]

        # Adjacent pairs NAC_{i, i+1}
        for i in range(1, nst - 1):
            val = abs(hvib[i, i+1]) * 1000.0 * units.au2ev
            x = MATRIX(1, 1)
            x.set(0, 0, val)
            nac.append(x)

    bin_supp, dens, cum = data_stat.cmat_distrib(nac, 0, 0, 0, 0, 30, 0.1)
    dens = np.array(dens)
  
    # Normalization of PD
    
    bin_width = 0.1  # same used in cmat_distrib
    area = np.sum(dens * bin_width)

    if area > 0:
        dens = dens / area  # Normalize so ∫PD dx = 1

    color = custom_colors.get(system, get_color(idx))
    plt.plot(bin_supp, dens, label=system, linewidth=2.5, color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-1, 1e2])

ax = plt.gca()

ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10) * 0.1, numticks=10))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xticks([1e-1, 1e0, 1e1, 1e2])
ax.set_xticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])

plt.xlabel('|NAC (S$_i$, S$_{i\pm1}$)| (meV)', fontsize=24)
plt.ylabel('PD (1/meV)', fontsize=24)
plt.legend(fontsize=16, ncol=3, loc='lower right', frameon=False)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.savefig('fig_b_NAC_adjacent_excited_states.png', dpi=600, bbox_inches='tight')
plt.show()


# In[ ]:




