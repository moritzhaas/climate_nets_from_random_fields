
import os, fnmatch, pickle
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
from collections import Counter
import networkx as nx
start_time = time.time()
#irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
#print('Task: ', irun)
curr_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
#corr_method='spearman'
#weighted = False # iterate over
#ranks = False
# if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
#     ranks = True

if not os.path.exists(base_path+'fdr/'):
    os.mkdir(base_path+'fdr/')

ar = 0
ar2 = None
var = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5
len_scale = 0.1

denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]
ks = [6, 60, 300, 600]#,1200]
#robust_tolerance = 0.5

filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

# %%
#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99
robust_tolerance = 0.2

if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')

# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins
#true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
# %%
# # load empcorrs, construct graphs, calc graph stats and save them
curr_time = time.time()
# empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
# empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
#for w in [True,False]:
#igrf_ES_matern_nu0.5_len0.1_ar0_fekete72_time1000_var10_seed1643248866.txt

for corr_method in ['BI-KSG','pearson', 'spearman','LWlin', 'binMI','HSIC', 'LW', 'ES']: # ['pearson', 'spearman', 'binMI', 'LW', 'ES', 'HSIC','BI-KSG']:
    print('Starting ', corr_method, time.time()-curr_time)
    curr_time = time.time()
    if corr_method == 'ES':
        empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_*',base_path+'empcorrs/')
        empdatas = find(f'*_nu{nu}_{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_*',base_path+'empdata/')
    elif corr_method == 'BI-KSG':
        empcorrs = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')]
        empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
    else:
        empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
        empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
    #outfilename = f'graphstats_{distrib}_{corr_method}_{typ}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
    all_fdr = np.zeros((numth))
    if irun < len(empcorrs):
        emp_corr = myload(empcorrs[irun])
    else:
        if irun < len(empdatas):
            orig_data = myload(empdatas[irun])
        else:
            seed = int(time.time())
            np.random.seed(seed)
            orig_data = diag_var_process(ar_coeff, cov, n_time)
            mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',orig_data)
        data = np.sqrt(var) * orig_data
        # save with nam in name
        for j in range(len(lat)):
            data[:,j] -= orig_data[:,j].mean()
            data[:,j] /= orig_data[:,j].std()
        emp_corr = compute_empcorr(data, similarity=corr_method)
    for i, density in enumerate(denslist):
        adj = get_adj(emp_corr, density, weighted=False)
        # accumulate over iruns: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdr[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum() #np.logical_and(adj != 0, dists > maxdist).sum() / (adj != 0).sum()
        #all_fdr[i,irun] = np.logical_and(adj != 0, dists > maxdist).sum() / (adj != 0).sum()

    if corr_method == 'ES':
        outfilename = f'fdr/fdr_part{irun}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_{num_runs}_numdens{len(denslist)}.txt'
    else:
        outfilename = f'fdr/fdr_part{irun}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}_numdens{len(denslist)}.txt'
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_fdr, fp)

# %%
# import matplotlib.pyplot as plt
# plt.clf()
# from tueplots import bundles
# plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update({"figure.dpi": 150})

# fig,ax = plt.subplots()
# for i in range(len(denslist)):
#     ax.plot(truecorrbins[:-1], plink[i,:])
# %%
#knn_adj(np.abs(emp_corr), ks[-5],weighted=False).mean()


# %%
# try variogram estimator
corr_method='pearson'
irun = 0
empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
#outfilename = f'graphstats_{distrib}_{corr_method}_{typ}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
all_fdr = np.zeros((numth))
if irun < len(empcorrs):
    emp_corr = myload(empcorrs[irun])
else:
    if irun < len(empdatas):
        orig_data = myload(empdatas[irun])
    else:
        seed = int(time.time())
        np.random.seed(seed)
        orig_data = diag_var_process(ar_coeff, cov, n_time)
        mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',orig_data)
    data = np.sqrt(var) * orig_data
    # save with nam in name
    for j in range(len(lat)):
        data[:,j] -= orig_data[:,j].mean()
        data[:,j] /= orig_data[:,j].std()
    emp_corr = compute_empcorr(data, similarity=corr_method)
for i, density in enumerate(denslist):
    adj = get_adj(emp_corr, density, weighted=False)
    # accumulate over iruns: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
    distadj = get_adj(5-dists, density,weighted=True)
    maxdist = 5-distadj[distadj!=0].min()
    all_fdr[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum() #np.logical_and(adj != 0, dists > maxdist).sum() / (adj != 0).sum()

# %%
# get Euclidean coordinates
n_time, num_points = data.shape

coords = np.zeros((n_time*num_points,3))
values = data.T.flatten()

for ipt in range(num_points):
    lo = grid.grid['lon'][ipt]
    la = grid.grid['lat'][ipt]
    coo = spherical2cartesian(lo, la)
    for i in range(n_time):
        coords[int(n_time*ipt)+i,:] = coo

# %%
# input to variogram estimator
rd_idcs = np.random.randint(0,num_points*n_time,10000)


#kernel crashes for long rd_idcs :(
import skgstat as skg
V = skg.Variogram(coords[rd_idcs,:], values[rd_idcs])
V.plot()
# %%
plt.plot(V.bins, V.experimental, '.b')

# for ipt in range(num_points):
#     for jpt in range(num_points):
#         # if dist in bin, predict variogram cov.