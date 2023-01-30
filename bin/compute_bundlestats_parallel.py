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


irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', irun)
start_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
corr_method='BI-KSG'
weighted = False
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True


ar = 0
ar2 = None
var = 10
robust_tolerance = 0.5
n_perm = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 0.5 # irrelevant
len_scale = 0.1 # irrelevant

denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]


filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

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
#true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
# %%
# # load empcorrs, construct graphs, calc bundle stats and save them
llquant = {}
dens = {}
tele = {}
robust_tele = {}

#%%
nbhds2, nbhds3 = {}, {}

nbhdname2 = f'nbhds_{grid_type}{num_points}_eps{eps2}.txt'
if not os.path.exists(base_path+'nbhds/'+nbhdname2):
    for i in range(num_points):
        nbhds2[i] = np.where(dists[i,:] <= eps2)[0]
    mysave(base_path+'nbhds/',nbhdname2, nbhds2)
else:
    nbhds2 = myload(base_path+'nbhds/'+nbhdname2)

nbhdname3 = f'nbhds_{grid_type}{num_points}_eps{eps3}.txt'
if not os.path.exists(base_path+'nbhds/'+nbhdname3):
    for i in range(num_points):
        nbhds3[i] = np.where(dists[i,:] <= eps3)[0]
    mysave(base_path+'nbhds/',nbhdname3, nbhds3)
else:
    nbhds3 = myload(base_path+'nbhds/'+nbhdname3)


# for file in os.listdir(base_path):
# if fnmatch.fnmatch(file, f'{distrib}*{runs}runs*') and fnmatch.fnmatch(file, f'*{corr_method}*fullrange_allstats.txt'):


# %%
curr_time = time.time()
for nu in nus:
    for len_scale in len_scales:
        print('Getting true covariance matrix.')
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
            cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
        else:
            cov = kernel(spherical2cartesian(lon,lat))
            mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
        if corr_method == 'BI-KSG':
            empcorrs = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')]
        else:
            empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
        empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
        outfilename = f'telestats_part{irun}_{distrib}_{corr_method}_{typ}_w{weighted}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}_tol{robust_tolerance}_eps{eps2}.txt'
        all_tele1 = np.zeros((numth))
        all_tele2 = np.zeros((numth))
        #all_tele3 = np.zeros((numth, num_runs))
        all_robusttele2 = np.zeros(numth)
        all_robusttele2raw = np.zeros((numth))
        #all_robusttele3 = np.zeros((numth, num_runs))
        # all_llquant1 = np.zeros((numth, num_runs))
        # all_llquant2 = np.zeros((numth, num_runs))
        all_mad = np.zeros((numth))
        all_shufflemad = np.zeros((numth,n_perm))
        all_dens = np.zeros((numth))
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
                print(f'{irun}, {outfilename}')
                continue
            else:
                emp_corr = compute_empcorr(data, similarity=corr_method)
        for i, density in enumerate(denslist):
            print('Computing tele. run, idens, time since previous print: ',irun,i, time.time()-curr_time)
            curr_time = time.time()
            if typ == 'threshold':
                adj = get_adj(emp_corr, density,weighted=weighted)
            else:
                adj = knn_adj(np.abs(emp_corr),ks[i],weighted = weighted)
            if weighted and ranks:
                adj = rank_matrix(adj)
                adj /= (adj.max()+1)
            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            lls = dists * adj
            all_tele1[i] = teleconn(adj, dists, 1e-5)
            all_tele2[i] = teleconn(adj, dists, eps2)
            #all_tele3[i] = teleconn(adj, dists, eps3)
            
            print('Computing robust tele. time since previous print: ',irun,i, time.time()-curr_time)
            curr_time = time.time()
            all_robusttele2[i] = robust_teleconn_nbhd(adj, dists, nbhds2,tolerance=robust_tolerance)
            all_robusttele2raw[i] =  robust_teleconn_nbhd(adj, dists, nbhds2, raw=True,tolerance=robust_tolerance)
            #all_robusttele3[i] = robust_teleconn_nbhd(adj, dists, nbhds3)
            all_dens[i] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
            
            print('Computing maxavgdeg. time since previous print: ',irun,i, time.time()-curr_time)
            curr_time = time.time()
            maxavgd, shufflexmaxdegs = maxavgdegbias(adj,nbhds2,num_perm=10)
            all_mad[i] = maxavgd
            all_shufflemad[i,:] = shufflexmaxdegs
        all_stats = [all_dens,all_tele1,all_tele2, all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad] # all_tele3, all_robusttele3,
        mysave(base_path+ 'bundlestats/',outfilename,all_stats)
        # all_stats = [all_dens,all_tele1,all_tele2, all_robusttele2, all_mad, all_shufflemad] # all_tele3, all_robusttele3,
        # with open(base_path + outfilename, "wb") as fp:   #Pickling
        #     pickle.dump(all_stats, fp)


# %%
#plt.plot(denslist, all_mad.mean(axis=1))
#plt.plot(denslist, all_shufflemad.mean(axis=1).mean(axis=1))
# %%
# #calculate_allstats(datalist, outfilename, typ, ks, dists, eps2, nbhds2, robust_tolerance, similarity = None, weighted=False, ranks=False, exp = False, alpha1 = 0.95, alpha2=0.99, var = 10)

# def calculate_allstats(filelist, outfilename, typ, ks, dists, eps, nbhds, robust_tolerance, similarity = None, weighted=False, ranks=False,exp = False, alpha1 = 0.95, alpha2=0.99, var = 10):
#     # when similarity is None, expect filelist to contain empcorrs, else expect empdata and compute empcorrs
#     # ks==denslist for typ==threshold
#     if not os.path.exists(base_path + outfilename):
#         num_runs = len(filelist)
#         all_degs, all_lls, all_ccs, all_spls = [np.zeros((numth, num_bins, num_runs)) for _ in range(4)]
#         all_dens, all_llquant1, all_llquant2, all_tele1, all_tele2, all_robusttele2, all_mad = [np.zeros((numth, num_runs)) for _ in range(7)]
#         all_spls = np.zeros((numth, len(splbins)-1, num_runs))
#         all_shufflemad = np.zeros((numth, 100, num_runs))
#         for run, nam in enumerate(filelist):
#             if similarity is None:
#                 emp_corr = myload(nam)
#             else:
#                 orig_data = myload(nam)
#                 nam = nam.split('data_',1)[1]
#                 data = np.sqrt(var) * orig_data
#                 exp_data = np.exp(np.sqrt(var) * orig_data)
#                 data2 = exp_data
#                 # save with nam in name
#                 for j in range(orig_data.shape[1]):
#                     data[:,j] -= orig_data[:,j].mean()
#                     data[:,j] /= orig_data[:,j].std()
#                     data2[:,j] -= exp_data[:,j].mean()
#                     data2[:,j] /= exp_data[:,j].std()
                
#                 if exp:
#                     emp_corr = compute_empcorr(data2, similarity)
#                 else:
#                     emp_corr = compute_empcorr(data, similarity)
#             for i in range(len(ks)):
#                 if typ == 'knn':
#                     k = ks[i]
#                     adj = knn_adj(np.abs(emp_corr),k, weighted = weighted)
#                 elif typ == 'threshold':
#                     density = ks[i]
#                     adj = get_adj(emp_corr, density, weighted=weighted)
#                 if ranks:
#                     adj = rank_matrix(adj)
#                     adj /= adj.max()
#                 deg = adj.sum(axis = 0)/(adj.shape[0]-1)
#                 all_dens[i,run] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))    

#                 #print('Avg. norm. deg of higher AR vs avg. norm. deg', deg[rd_idcs].mean(), deg.mean())
#                 #ar_bias[i,run] = deg[rd_idcs].mean() / deg.mean()

#                 print('Computing graphstats. ', time.time()-curr_time)
#                 curr_time = time.time()
#                 # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
#                 degnum = np.histogram(deg, bins=degbins)[0]
#                 all_degs[i,:,run] = degnum/num_points
                
#                 G = nx.from_numpy_matrix(adj)
#                 #idcs = np.random.permutation(adj.shape[0])[:n_estim]
#                 # if weighted:
#                 #     ccw = nx.clustering(G, weight = 'weight')
#                 #     ccwnum = np.histogram(list(ccw.values()),bins = degbins)[0]
#                 #     all_ccws[i,:,run] = ccwnum/num_points
#                 #     #or spl for weighted:
#                 #     #all_spaths = nx.shortest_path(G)
#                 #     #splw = np.zeros_like(all_spaths) # fix because dicts
#                 #     #for i in range(all_spaths.shape[0]):
#                 #     #    for j in range(all_spaths.shape[1]):
#                 #     #        splw[i,j] = len(all_spaths[i][j])-1
#                 # else:
#                 #     betwe = nx.betweenness_centrality(G, k = n_estim)
#                 #     betwnum = np.histogram(list(betwe.values()), bins = betwbins)[0]
#                 #     all_betw[i,:,run] = betwnum/num_points

#                 cc = nx.clustering(G) #, nodes = idcs)
#                 spl = dict(nx.all_pairs_shortest_path_length(G))
#                 thesespl = []
#                 for ipt in range(num_points):
#                     for jpt in list(spl[ipt].keys()):
#                         if jpt != ipt:
#                             thesespl.append(spl[ipt][jpt])
#                 ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
#                 splnum = np.histogram(thesespl,bins = splbins)[0] / 2
                
#                 all_ccs[i,:,run] = ccnum/num_points
#                 all_spls[i,:,run] = splnum/num_points
#                 # accumulate over runs: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
#                 for ibin in range(n_dresol):
#                     if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
#                         plink[i,ibin] += (adj[np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1])] != 0).sum()
#                     else:
#                         plink[i,ibin] = np.nan
                
#                 print('Computing length stats. ', time.time()-curr_time)
#                 curr_time = time.time()
#                 all_tele1[i,run] = teleconn(adj, dists, 1e-8)
#                 all_tele2[i,run] = teleconn(adj, dists, eps)
#                 all_robusttele2[i,run] = robust_teleconn_nbhd(adj, dists, nbhds, tolerance = robust_tolerance)
#                 maxavgdeg, shufflexmaxdegs = maxavgdegbias(trueadj,nbhds)
#                 all_mad[i,run] = maxavgdeg
#                 all_shufflemad[i,:,run] = shufflexmaxdegs
#                 lls = dists * (adj != 0)
#                 if weighted:
#                     lld = lldistr(dists,adj,llbins)
#                     all_lls[i,:,run] = lld
#                     # where does weight of all edges <= length surpass (alpha1*total_weight)
#                     total_weight = adj.sum()/2
#                     cum_lld = np.zeros_like(lld)
#                     cum_lld[0] = lld[0]
#                     for j in range(len(lld)-1):
#                         cum_lld[j+1] = cum_lld[j] + lld[j+1]
#                     all_llquant1[i, run] = llbins[np.where(cum_lld >= alpha1*total_weight)[0][0]]
#                     all_llquant2[i, run] = llbins[np.where(cum_lld >= alpha2*total_weight)[0][0]]
#                 else:
#                     sorted_lls = np.sort((lls)[lls != 0])
#                     all_lls[i,:,run] = np.histogram(sorted_lls, bins = llbins)[0]
#                     if len(sorted_lls) == 0:
#                         all_llquant1[i, run] = 0
#                         all_llquant2[i, run] = 0
#                     else:
#                         all_llquant1[i, run] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
#                         all_llquant2[i, run] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]
#         # after all runs
#         for ibin in range(n_dresol):
#             if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
#                 plink[:,ibin] /= (num_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())
#         all_stats = [all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad]
#         with open(base_path + outfilename, "wb") as fp:   #Pickling
#             pickle.dump(all_stats, fp)
#             #np.save(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy', true_stats)
#     else:
#         with open(base_path + outfilename, "rb") as fp:   # Unpickling
#             all_stats = pickle.load(fp)
#             # true_stats = np.load(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy')
#         all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad = all_stats
#     return all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad
