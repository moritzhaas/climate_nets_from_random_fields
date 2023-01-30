import os, fnmatch, pickle
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
from collections import Counter
import networkx as nx
start_time = time.time()

irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', irun)
base_path = '../../climnet_output/'
distrib = 'igrf'


typ = 'threshold'
grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='pearson' #'pearson'
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True


ar = 0
ar2 = None
var = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 0.5
len_scale = 0.2

threshs = [0.1,0.2,0.3,0.5,0.8]
denslist = [0.001,0.01,0.05,0.1]#0.2
ks = [6, 60, 300, 600]#1200
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
robust_tolerance = 0.5
degbinssparse = np.linspace(0,0.5, num_bins+1)
degbinsw = np.linspace(0,0.2, num_bins+1)



if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')

# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins
#true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad

# %%
all_dens = np.zeros((len(nus),len(len_scales),len(threshs)))
all_lengths1,all_lengths2 = [np.zeros((len(nus),len(len_scales),len(threshs),len(dists))) for _ in range(2)]
avgcorrmean,avgcorrstd = [np.zeros((len(nus),len(len_scales))) for _ in range(2)]

# # load empcorrs, construct graphs, calc graph stats and save them
curr_time = time.time()
for inu,nu in enumerate(nus):
    for ilen, len_scale in enumerate(len_scales):
        if corr_method == 'BI-KSG':
            empcorrs = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')]
        else:
            empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
        empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
        #for w in [True,False]:
    
        # if typ == 'knn' and not w:
        #     continue
        
        all_degs, all_lls,all_degws, all_llws, all_ccs,all_ccws, all_spls = [np.zeros((numth, num_bins)) for _ in range(7)]
        all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = [np.zeros((numth)) for _ in range(4)]

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
            if corr_method == 'BI-KSG':
                raise FileNotFoundError(f'{irun}, outfilename')
            else:
                emp_corr = compute_empcorr(data, similarity=corr_method)
        
        for ithres,thresh in enumerate(threshs):
            adj = emp_corr > thresh
            adj[np.eye(adj.shape[0],dtype=bool)] = 0
            density = adj.sum()/((adj.shape[0]-1)*adj.shape[0])
            all_dens[inu,ilen,ithres] = density
            # distadj = get_adj(5-dists, density,weighted=True)
            # maxdist = 5-distadj[distadj!=0].min()
            all_lengths1[inu,ilen,ithres] = decorr_length(adj,dists,min_connectivity=0.8, grid_type=grid_type,base_path=base_path).mean()#bin then save
            all_lengths2[inu,ilen,ithres] = decorr_length(adj,dists,min_connectivity=0.5, grid_type=grid_type,base_path=base_path).mean()
        thisavgcorr = np.zeros(num_points)
        for pt in range(num_points):
            thisavgcorr[pt] = emp_corr[pt,nbhds2[pt]][emp_corr[pt,nbhds2[pt]]!=1].mean()
        avgcorrmean[inu,ilen] = thisavgcorr.mean()
        avgcorrstd[inu,ilen] = thisavgcorr.std()
            
outfilename = f'rfstats/rfstats_part{irun}_{distrib}_{corr_method}_{typ}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'    
all_stats = [all_dens,avgcorrmean,avgcorrstd,all_lengths1,all_lengths2]
with open(base_path + outfilename, "wb") as fp:   #Pickling
    pickle.dump(all_stats, fp)



# %%
# %%
from sklearn.gaussian_process.kernels import Matern
nus = [0.5,1.5]
len_scales = [0.1,0.2]
all_denstrue = np.zeros((len(nus),len(len_scales),len(threshs)))
all_lengths1true,all_lengths2true = np.zeros((len(nus),len(len_scales),len(threshs),len(dists))),np.zeros((len(nus),len(len_scales),len(threshs),len(dists)))
if not os.path.exists(base_path+f'decorrlen_migrfs_th{threshs}.txt'):
    for inu,nu in enumerate(nus):
        for ilen,len_scale in enumerate(len_scales):
            #compute cov
            kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
            cov = kernel(spherical2cartesian(lon,lat))
            
            # compute decorr length
            for ithres,thresh in enumerate(threshs):
                adj = cov > thresh
                adj[np.eye(adj.shape[0],dtype=bool)] = 0
                density = adj.sum()/((adj.shape[0]-1)*adj.shape[0])
                all_denstrue[inu,ilen,ithres] = density
                # distadj = get_adj(5-dists, density,weighted=True)
                # maxdist = 5-distadj[distadj!=0].min()
                all_lengths1true[inu,ilen,ithres,:] = decorr_length(adj,dists,min_connectivity=0.8, grid_type=grid_type,base_path=base_path)#bin then save
                all_lengths2true[inu,ilen,ithres,:] = decorr_length(adj,dists,min_connectivity=0.5, grid_type=grid_type,base_path=base_path)

    mysave(base_path,f'decorrlen_migrfs_th{threshs}.txt',[all_denstrue,all_lengths1true,all_lengths2true])
else:
    all_denstrue,all_lengths1true,all_lengths2true = myload(base_path+f'decorrlen_migrfs_th{threshs}.txt')
# %%
ithres = 1
thresh=threshs[ithres]
print(f'Empirical nets, thresh={thresh}')

all_denss,avgcorrmeans,avgcorrstds,all_lengths1s,all_lengths2s = myload(find(f'allrfstats*','../../climnet_output/'+'rfstats/')[0])
#[inu,ilen,ithres,irun]
for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        #for ithres,thresh in enumerate(threshs):
        print(nu,len_scale,all_denss[inu,ilen,ithres,:].mean(), all_lengths1s[inu,ilen,ithres,:].mean())


# compare to clim variables
print('True nets')
for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        #for ithres,thresh in enumerate(threshs):
        print(nu,len_scale,all_denstrue[inu,ilen,ithres], all_lengths1true[inu,ilen,ithres,:].mean())

decorrstats = {}
decorrstatsnoboot = {}
for var_name in var_names:
    if var_name=='z250' or var_name=='z850':
        continue
    decorrstatsnoboot[var_name] = myload(base_path+f'decorrstats_noboot_{var_name}_dtele{dtele}_{corr_method}_w{weighted}_seed{seed}.txt')
    #for ithres,thresh in enumerate(threshs):
    print(var_name, decorrstatsnoboot[var_name][0][ithres],decorrstatsnoboot[var_name][2][ithres,:].mean())


# %%
ithres = 1
thresh=threshs[ithres]
print(f'Empirical nets, thresh={thresh}')

all_denss,avgcorrmeans,avgcorrstds,all_lengths1s,all_lengths2s = myload(find(f'allrfstats*','../../climnet_output/'+'rfstats/')[0])
#[inu,ilen,ithres,irun]
for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        #for ithres,thresh in enumerate(threshs):
        print(nu,len_scale,avgcorrmeans[inu,ilen,:].mean(), avgcorrstds[inu,ilen,:].mean())


