# %%
import os, fnmatch, pickle
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
#from multiprocessing import Pool
from collections import Counter
from sklearn.gaussian_process.kernels import Matern


#irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
start_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
corr_method='spearman'
weighted = False
ranks = False

ar = 0
ar2 = None
var = 10
robust_tolerance = 0.2
n_perm = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 0.5 # irrelevant
len_scale = 0.1 # irrelevant

denslist = [0.001,0.01,0.05,0.1,0.2]#,0.2]
ks = [6, 60, 300, 600,1200]#,1200]


#filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

# %%
#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99


if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')

# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points


# %%
num_points = gridstep_to_numpoints(2.5)
for len_scale in len_scales:
    for nu in nus:
        hyperlist = []
        for filename in find(f'empcorrdict_*_part0_BI-KSG_*nu{nu}_len{len_scale}*', base_path+ 'empcorrs/',nosub=True):
            hyperparams = filename.split('_part0_BI-KSG_',10)[-1]
            if hyperparams not in hyperlist:
                hyperlist.append(hyperparams)

        for hyperparams in hyperlist:
            for irun in range(num_runs):
                these_files = list(set(find(f'*run{irun}_part*'+hyperparams, base_path+ 'empcorrs/')) & set(find('empcorrdict_*', base_path+ 'empcorrs/',nosub=True)))
                these_runs = len(these_files)
                if these_runs < num_runs:
                    print(f'Run {irun} incomplete, only {these_runs}.')
                    try:
                        print(these_files[0])
                    except:
                        print(irun)
                    continue
            # runs = []
            # for filename in all_files:
            #     irun = filename.split('_run',1)[1].split('_',10)[0]
            #     if irun not in runs:
            #         runs.append(irun)
            # for irun in runs:
            #     these_files = [nam  for nam in all_files if irun == nam.split('_run',1)[1].split('_',10)[0]]
            #     these_runs = len(these_files)
            #     if these_runs<num_runs:
            #         print(hyperparams+' has only '+ str(these_runs))
            #     elif these_runs > num_runs:
            #         print(hyperparams+' has even '+ str(these_runs))
                try:
                    empcorr_dict = myload(these_files[0])
                except:
                    print('Too many variables. ',these_files[0])
                    continue
                emp_corr = np.zeros((num_points,num_points))
                for nam in these_files:
                    empcorr_dict = myload(nam)
                    idcs = list(empcorr_dict.keys())
                    corrvals = list(empcorr_dict.values())
                    for i in range(len(idcs)):
                        emp_corr[idcs[i]] = corrvals[i]
                emp_corr[np.triu(np.ones_like(emp_corr), k = 1).T != 0] = emp_corr[np.triu(np.ones_like(emp_corr), k = 1) != 0]

                outfilename = nam.split('_part',1)[0].split('empcorrs/',10)[-1] + '_BI-KSG_' + hyperparams
                mysave(base_path+ 'empcorrs/',outfilename,emp_corr)
                #if not os.path.exists(base_path+'empcorrs/old/'):
                #    os.mkdir(base_path+'empcorrs/old/')
                for nam in these_files:
                    filenam = nam.split('empcorrs/',1)[1]
                    os.remove(nam) #os.replace(nam, base_path+'empcorrs/old/'+filenam)

# %%
for len_scale in len_scales:
    for nu in nus:
        for irun in range(30):
            theseempcorrs = find(f'empcorrdict_run{irun}_BI-KSG_matern_nu{nu}_len{len_scale}_ar0_fekete72_var10_*',base_path+'empcorrs/',nosub=True)
            if theseempcorrs != []:
                try:
                    emp_corr = myload(theseempcorrs[0])
                    if emp_corr.shape != (num_points,num_points):
                        raise ValueError('BIKSG has wrong shape!')
                    emp_corr[np.triu(np.ones_like(emp_corr), k = 1).T != 0] = emp_corr[np.triu(np.ones_like(emp_corr), k = 1) != 0]
                
                    for nam in find(f'empcorrdict_run{irun}_part*_BI-KSG_matern_nu{nu}_len{len_scale}_ar0_fekete72_var10_*',base_path+'empcorrs/'):
                        os.remove(nam)
                except:
                    print(f'{len_scale} {nu} {irun} has not saved array.')



# %%
for nu in [0.5,1.5]:
    for len_scale in [0.1,0.2]:
        print(f'nu={nu}, len={len_scale}: ')
        numfiles = []
        for irun in range(30):
            numfiles.append(len(find(f'empcorrdict_run{irun}_part*_BI-KSG_*nu{nu}_len{len_scale}*', base_path+ 'empcorrs/',nosub=True)))
        print(numfiles)


# %%
from sklearn.gaussian_process.kernels import Matern
xs = np.linspace(0,180,100)
lowcorr = 0.05
print(f'Distance in km at which correlation below {lowcorr}')
for nu in nus:
    for len_scale in len_scales:
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        ys = kernel(spherical2cartesian(xs,np.zeros_like(xs)))[0,:]
        #plt.plot(xs, ys)
        print(nu, len_scale, xs[np.where(ys < lowcorr)[0][0]] * np.pi / 180 * earth_radius)

# %%
# biksg values not >= 0, but for this estimator also lots of 0 truncated values..
from sklearn.feature_selection import _mutual_info, mutual_info_regression
n = 1000
orig_data = diag_var_process(ar_coeff[:n], cov[:n,:n], n_time)
mi=mutual_info_regression(orig_data[:,:-1],orig_data[:,-1],n_neighbors=5)

(mi==0).sum()