
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

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='BI-KSG' #'pearson'
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
len_scales = [0.05,0.1,0.2]
nu = 0.5
len_scale = 0.2

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
# # load empcorrs, construct graphs, calc graph stats and save them
curr_time = time.time()
if corr_method == 'BI-KSG':
    empcorrs = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')]
else:
    empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
#for w in [True,False]:
for typ in ['threshold','knn']:
    # if typ == 'knn' and not w:
    #     continue
    outfilename = f'graphstats_part{irun}_{distrib}_{corr_method}_{typ}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
    all_degs, all_lls,all_degws, all_llws, all_ccs,all_ccws, all_spls = [np.zeros((numth, num_bins)) for _ in range(7)]
    all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = [np.zeros((numth)) for _ in range(4)]
    all_spls = np.zeros((numth, len(splbins)-1))
    all_dens = np.zeros((numth))
    all_densw = np.zeros((numth))
    plink = np.zeros((numth, n_dresol))
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
    for i, density in enumerate(denslist):
        if typ == 'threshold':
            adj = get_adj(emp_corr, density, weighted=False)
        else:
            k = ks[i]
            adj = knn_adj(np.abs(emp_corr),k,weighted=False)
        deg = adj.sum(axis = 0)/(adj.shape[0]-1)
        all_dens[i ] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
        print('Computing degd and lld., typ, densidx',typ,i, time.time()-curr_time)
        curr_time = time.time()
        # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
        degnum = np.histogram(deg, bins=degbinssparse)[0]
        all_degs[i,: ] = degnum/num_points
        lls = dists * (adj != 0)
        sorted_lls = np.sort((lls)[lls != 0])
        all_lls[i,: ] = np.histogram(sorted_lls, bins = llbins)[0]
        if len(sorted_lls) == 0:
            all_llquant1[i ] = 0
            all_llquant2[i ] = 0
        else:
            all_llquant1[i ] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
            all_llquant2[i ] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]
        print('Computing ccd, spld. and plink. Previous part took:', time.time()-curr_time)
        curr_time = time.time()
        G = nx.from_numpy_matrix(adj)
        cc = nx.clustering(G) #, nodes = idcs)
        spl = dict(nx.all_pairs_shortest_path_length(G))
        thesespl = []
        for ipt in range(num_points):
            for jpt in list(spl[ipt].keys()):
                if jpt != ipt:
                    thesespl.append(spl[ipt][jpt])
        ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
        splnum = np.histogram(thesespl,bins = splbins)[0] / 2
        
        all_ccs[i,: ] = ccnum/num_points
        all_spls[i,: ] = splnum/num_points
        # accumulate over iruns: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
        for ibin in range(n_dresol):
            if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
                plink[i,ibin] += (adj[np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1])] != 0).sum()
            else:
                plink[i,ibin] = np.nan
        
        # start weighted graphs
        if typ == 'threshold':
            adj = get_adj(emp_corr, density, weighted=True)
        else:
            k = ks[i]
            adj = knn_adj(np.abs(emp_corr),k,weighted=True)
        if ranks: # use for MI..
            adj = rank_matrix(adj)
            adj /= (adj.max()+1)
        deg = adj.sum(axis = 0)/(adj.shape[0]-1)
        all_densw[i ] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
        print('Computing degd and lld., typ, densidx',typ,i, time.time()-curr_time)
        curr_time = time.time()
        # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
        degnum = np.histogram(deg, bins=degbinsw)[0]
        all_degws[i,: ] = degnum/num_points
        lls = dists * (adj != 0)
        lld = lldistr(dists,adj,llbins)
        all_llws[i,: ] = lld
        # where does weight of all edges <= length surpass (alpha1*total_weight)
        total_weight = adj.sum()/2
        cum_lld = np.zeros_like(lld)
        cum_lld[0] = lld[0]
        for j in range(len(lld)-1):
            cum_lld[j+1] = cum_lld[j] + lld[j+1]
        all_llwquant1[i ] = llbins[np.where(cum_lld >= alpha1*total_weight)[0][0]]
        all_llwquant2[i ] = llbins[np.where(cum_lld >= alpha2*total_weight)[0][0]]
        
        print('Computing ccd, spld. and plink. Previous part took:', time.time()-curr_time)
        curr_time = time.time()
        G = nx.from_numpy_matrix(adj)
        ccw = nx.clustering(G, weight = 'weight')
        ccwnum = np.histogram(list(ccw.values()),bins = degbins)[0]
        all_ccws[i,: ] = ccwnum/num_points
            
    # # after all runs
    # for ibin in range(n_dresol):
    #     if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
    #         plink[:,ibin] /= (num_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())

    all_stats = [all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2]
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_stats, fp)

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
plt.hist(upper(dists)[upper(dists) <=0.5],100)

