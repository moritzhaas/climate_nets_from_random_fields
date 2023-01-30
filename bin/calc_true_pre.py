# %%
import os, fnmatch, pickle
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
from sklearn.gaussian_process.kernels import Matern

grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
#corr_method='spearman'
nus = [1.5,0.5]
len_scales = [0.1,0.2]
nu = 0.5 # only for grid helper
len_scale = 0.1
ranks = False
#if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
#    ranks = True
num_runs = 30
n_time = 100

denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]
robust_tolerance = 0.2

numth = len(denslist)

exec(open("grid_helper.py").read())

true_dens, true_densw, true_llquant1,true_llquant2,true_llwquant1,true_llwquant2 = [np.zeros((numth)) for _ in range(6)]

try: robust_tolerance
except NameError: robust_tolerance = None

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

if not os.path.exists(base_path+'truestats/'):
    os.mkdir(base_path+'truestats/')

degbinssparse = np.linspace(0,0.5, num_bins+1)
degbinsw = np.linspace(0,0.2, num_bins+1)

curr_time = time.time()
for nu in nus:
    for len_scale in len_scales:
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        cov = kernel(spherical2cartesian(lon,lat))
        # calculate ground truth for teleconn.
        for ty in ['threshold', 'knn']:
            true_degs, true_degws,true_lls,true_llws,true_ccs,true_ccws= [np.zeros((numth, num_bins)) for _ in range(6)]
            true_spls = np.zeros((numth, len(splbins)-1)) 
            truestats_file = f'truestats/truegraphstats_matern{nu}_{len_scale}_{ty}_{len(ks)}_ranks{ranks}.txt'
            if not os.path.exists(base_path + truestats_file):
                #bothtele = np.zeros((len(ks),2))
                for i in range(len(ks)):
                    if ty == 'knn':
                        k = ks[i]
                        true_adj = knn_adj(np.abs(cov),k, weighted = False)
                    elif ty == 'threshold':
                        density = denslist[i]
                        true_adj = get_adj(cov, density, weighted = False)

                    deg = true_adj.sum(axis = 0)/(true_adj.shape[0]-1)
                    true_dens[i] = true_adj.sum()/ (true_adj.shape[0]*(true_adj.shape[0]-1))
                    print('Computing true stats',ty,i, time.time()-curr_time)
                    curr_time = time.time()
                    # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
                    degnum = np.histogram(deg, bins=degbinssparse)[0]
                    true_degs[i,: ] = degnum/num_points
                    lls = dists * (true_adj != 0)
                    sorted_lls = np.sort((lls)[lls != 0])
                    true_lls[i,: ] = np.histogram(sorted_lls, bins = llbins)[0]
                    if len(sorted_lls) == 0:
                        true_llquant1[i ] = 0
                        true_llquant2[i ] = 0
                    else:
                        true_llquant1[i ] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
                        true_llquant2[i ] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]
                    print('Computing ccd, spld. Previous part took:', time.time()-curr_time)
                    curr_time = time.time()
                    G = nx.from_numpy_matrix(true_adj)
                    cc = nx.clustering(G) #, nodes = idcs)
                    spl = dict(nx.all_pairs_shortest_path_length(G))
                    thesespl = []
                    for ipt in range(num_points):
                        for jpt in list(spl[ipt].keys()):
                            if jpt != ipt:
                                thesespl.append(spl[ipt][jpt])
                    ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
                    splnum = np.histogram(thesespl,bins = splbins)[0] / 2
                    
                    true_ccs[i,: ] = ccnum/num_points
                    true_spls[i,: ] = splnum/num_points
                    
                    # if all_robusttele2 is not None:
                    #     true_tele1[i] = teleconn(true_adj, dists, 1e-8)
                    #     true_tele2[i] = teleconn(true_adj, dists, eps2)
                    #     true_robusttele2[i] = robust_teleconn_nbhd(true_adj, dists, nbhds2, tolerance = robust_tolerance)
                    #     balldeg, shufflexmaxdegs = maxavgdegbias(true_adj,nbhds2)
                    #     true_mad[i] = balldeg
                    #     true_shufflemad[i,:] = shufflexmaxdegs

                    # start weighted graphs
                    if ty == 'threshold':
                        wadj = get_adj(cov, density, weighted=True)
                    else:
                        k = ks[i]
                        wadj = knn_adj(np.abs(cov),k,weighted=True)
                    if ranks: # use for MI..
                        wadj = rank_matrix(wadj)
                        wadj /= (wadj.max()+1)
                    deg = wadj.sum(axis = 0)/(wadj.shape[0]-1)
                    true_densw[i ] = wadj.sum()/ (wadj.shape[0]*(wadj.shape[0]-1))
                    print('Computing weighted truestats',ty,i, time.time()-curr_time)
                    curr_time = time.time()
                    # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
                    degnum = np.histogram(deg, bins=degbinsw)[0]
                    true_degws[i,: ] = degnum/num_points
                    lls = dists * (wadj != 0)
                    lld = lldistr(dists,wadj,llbins)
                    true_llws[i,: ] = lld
                    # where does weight of all edges <= length surpass (alpha1*total_weight)
                    total_weight = wadj.sum()/2
                    cum_lld = np.zeros_like(lld)
                    cum_lld[0] = lld[0]
                    for j in range(len(lld)-1):
                        cum_lld[j+1] = cum_lld[j] + lld[j+1]
                    true_llwquant1[i ] = llbins[np.where(cum_lld >= alpha1*total_weight)[0][0]]
                    true_llwquant2[i ] = llbins[np.where(cum_lld >= alpha2*total_weight)[0][0]]
                    
                    print('Computing ccw. Previous part took:', time.time()-curr_time)
                    curr_time = time.time()
                    G = nx.from_numpy_matrix(wadj)
                    ccw = nx.clustering(G, weight = 'weight')
                    ccwnum = np.histogram(list(ccw.values()),bins = degbins)[0]
                    true_ccws[i,: ] = ccwnum/num_points

                #true_stats = [true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2]
                true_stats = [true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2]#,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad]
                with open(base_path + truestats_file, "wb") as fp:   #Pickling
                    pickle.dump(true_stats, fp)
                    #np.save(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy', true_stats)
            else:
                with open(base_path + truestats_file, "rb") as fp:   # Unpickling
                    true_stats = pickle.load(fp)
                    # true_stats = np.load(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy')
                #true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_stats
            
            # Compute telestats.
            true_tele1,true_tele2,true_robusttele2,true_robusttele2raw, true_mad,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw, true_madw = [np.zeros((numth)) for _ in range(10)]
            true_shufflemad = np.zeros((numth,100))
            true_shufflemadw = np.zeros((numth,100))
            true_lls,true_llws = [np.zeros((numth,len(llbins)-1)) for _ in range(2)]
            # add more tele?
            for w in [True, False]:
                truestats_file = f'truestats/truetelestats_matern{nu}_{len_scale}_{ty}_{len(ks)}_tol{robust_tolerance}_eps{eps2}_w{w}_ranks{ranks}.txt'
                if not os.path.exists(base_path + truestats_file):
                    for i in range(len(ks)):
                        if ty == 'knn':
                            k = ks[i]
                            true_adj = knn_adj(np.abs(cov),k, weighted = w)
                        elif ty == 'threshold':
                            density = denslist[i]
                            true_adj = get_adj(cov, density, weighted = w)
                        if w and ranks:
                            true_adj = rank_matrix(true_adj)
                            true_adj /= (true_adj.max()+1)
                        deg = true_adj.sum(axis = 0)/(true_adj.shape[0]-1)
                        if w:
                            wadj = true_adj
                            true_densw[i] = true_adj.sum()/ (true_adj.shape[0]*(true_adj.shape[0]-1))
                        else:
                            true_dens[i] = true_adj.sum()/ (true_adj.shape[0]*(true_adj.shape[0]-1))
                        print('Computing true telestats',ty,i, time.time()-curr_time)
                        curr_time = time.time()
                        # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
                        lls = dists * (true_adj != 0)
                        sorted_lls = np.sort((lls)[lls != 0])
                        if w:
                            lld = lldistr(dists,wadj,llbins)
                            true_llws[i,: ] = lld
                            # where does weight of all edges <= length surpass (alpha1*total_weight)
                            total_weight = wadj.sum()/2
                            cum_lld = np.zeros_like(lld)
                            cum_lld[0] = lld[0]
                            for j in range(len(lld)-1):
                                cum_lld[j+1] = cum_lld[j] + lld[j+1]
                            true_llwquant1[i ] = llbins[np.where(cum_lld >= alpha1*total_weight)[0][0]]
                            true_llwquant2[i ] = llbins[np.where(cum_lld >= alpha2*total_weight)[0][0]]
                        else:
                            true_lls[i,: ] = np.histogram(sorted_lls, bins = llbins)[0]
                            if len(sorted_lls) == 0:
                                true_llquant1[i ] = 0
                                true_llquant2[i ] = 0
                            else:
                                true_llquant1[i ] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
                                true_llquant2[i ] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

                        if w:
                            true_telew1[i] = teleconn(true_adj, dists, 1e-8)
                            true_telew2[i] = teleconn(true_adj, dists, eps2)
                            true_robusttelew2[i] = robust_teleconn_nbhd(true_adj, dists, nbhds2, tolerance = robust_tolerance)
                            true_robusttelew2raw[i] = robust_teleconn_nbhd(true_adj, dists, nbhds2, tolerance = robust_tolerance,raw=True)
                            balldeg, shufflexmaxdegs = maxavgdegbias(true_adj,nbhds2)
                            true_madw[i] = balldeg
                            true_shufflemadw[i,:] = shufflexmaxdegs
                        else:
                            true_tele1[i] = teleconn(true_adj, dists, 1e-8)
                            true_tele2[i] = teleconn(true_adj, dists, eps2)
                            true_robusttele2[i] = robust_teleconn_nbhd(true_adj, dists, nbhds2, tolerance = robust_tolerance)
                            true_robusttele2raw[i] = robust_teleconn_nbhd(true_adj, dists, nbhds2, tolerance = robust_tolerance,raw=True)
                            balldeg, shufflexmaxdegs = maxavgdegbias(true_adj,nbhds2)
                            true_mad[i] = balldeg
                            true_shufflemad[i,:] = shufflexmaxdegs
                    if w:
                        true_stats = [true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw]
                    else:
                        true_stats = [true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad]
                    with open(base_path + truestats_file, "wb") as fp:   #Pickling
                        pickle.dump(true_stats, fp)
                        #np.save(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy', true_stats)
                else:
                    with open(base_path + truestats_file, "rb") as fp:   # Unpickling
                        if w:
                            true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2, true_robusttelew2raw, true_madw, true_shufflemadw = pickle.load(fp)
                        else:
                            true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = pickle.load(fp)

# %%