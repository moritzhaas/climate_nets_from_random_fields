import networkx as nx
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from climnet.grid import FeketeGrid, regular_lon_lat
from climnet.myutils import *
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']

# std parameters
# n_lat = 4*18
# num_runs = 2
# n_time = 100
# ar = 0
# ar2 = None
# distrib = 'igrf'
# grid_type = 'fekete' # regular, landsea
# typ = 'threshold' # 'knn' 'threshold'
# weighted = False
# ranks = False
# corr_method='spearman' # 'spearman', 'MI', 'HSIC', 'ES'
# robust_tolerance = 0.5
# denslist = [0.001,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
# ks = [6, 60, 300, 600,1200] #[5,  10, 65, 125, 250]

# if len(denslist) != len(ks) and typ == 'threshold':
#     raise RuntimeError('Denslist needs to have same length as ks.')

# # covariance parameters
# var = 10
# nu = 0.5
# len_scale = 0.1

alpha = 0.05
num_bins = 100
n_estim = 500
n_dresol = 100
numth = len(denslist) #51

force_run = True
reuse = False
save = True

# # distribution
# va = 0
# for l in range(len(A)):
#     va += A[l] * (2 * l + 1) / (4 * np.pi)
# A *= var/va

similarities = ['pearson', 'spearman', 'binMI', 'LW', 'BI-KSG', 'ES', 'HSIC']
base_path = '../../climnet_output/' #'~/Documents/climnet_output/'

# %%
# generate grid
n_lon = 2 * n_lat
grid_step_lon = 360/ n_lon
grid_step_lat = 180/ n_lat
dist_equator = gdistance((0,0),(0,grid_step_lon),radius=1)
lon2, lat2 = regular_lon_lat(n_lon,n_lat)
regular_grid = {'lon': lon2, 'lat': lat2}
start_date = '2000-01-01'
if os.path.exists(base_path + f'grids/regular_dists_nlat_{n_lat}_nlon_{n_lon}.txt'):
    reg_dists = myload(base_path +f'grids/regular_dists_nlat_{n_lat}_nlon_{n_lon}.txt')
else:
    reg_dists = all_dists(lat2,lon2)
    mysave(base_path+'grids/', f'regular_dists_nlat_{n_lat}_nlon_{n_lon}.txt', reg_dists)

# create fekete grid
num_points = gridstep_to_numpoints(grid_step_lon)
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
if os.path.exists(base_path + f'grids/fekete_dists_npoints_{num_points}.txt'):
    dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')
else:
    dists = np.zeros((len(lon), len(lon)))
    for i in range(len(lon)):
        for j in range(i):
            dists[i,j] = gdistance((lat[i], lon[i]), (lat[j],lon[j]), radius=1)
            dists[j,i] = dists[i,j]
    mysave(base_path+'grids/', f'fekete_dists_npoints_{num_points}.txt', dists)


earth_radius = 6371.009
num_nodes = dists.shape[0]

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99



degbins = np.linspace(0,1, num_bins+1)
degbinssparse = np.linspace(0,0.5, num_bins+1)
degbinsw = np.linspace(0,0.2, num_bins+1)
eigbins = np.linspace(0,0.3,num_bins+1)
betwbins = np.linspace(0,0.17, num_bins + 1)
thbins = np.linspace(0,1,numth)
llbins = np.linspace(0, dists.max(), num_bins+1)
splbins = np.concatenate((np.arange(0,11), np.linspace(11,num_points/4,9)))
truecorrbins = np.linspace(0,1,n_dresol+1)
all_degs = np.zeros((numth, num_bins, num_runs))
all_lls = np.zeros((numth, num_bins, num_runs))
all_ccs = np.zeros((numth, num_bins, num_runs))
all_ccws = np.zeros((numth, num_bins, num_runs))
all_spls = np.zeros((numth, len(splbins)-1, num_runs))
all_betw = np.zeros((numth, num_bins, num_runs))
all_eigc = np.zeros((len(denslist), num_bins, num_runs))
all_dens = np.zeros((numth, num_runs))
all_tele1 = np.zeros((numth, num_runs))
all_tele2 = np.zeros((numth, num_runs))
all_robusttele2 = np.zeros((numth, num_runs))
all_llquant1 = np.zeros((numth, num_runs))
all_llquant2 = np.zeros((numth, num_runs))
all_telequant = np.zeros((numth, num_runs))
all_telequant2 = np.zeros((numth, num_runs))
all_mad = np.zeros((numth, num_runs))
all_shufflemad = np.zeros((numth,100,num_runs))
plink = np.zeros((numth, n_dresol))

ar_bias = np.zeros((numth, num_runs))

# if numth <= 35:
#     ks = np.linspace(1,numth,numth, dtype=int)
# else:
#     ks = np.concatenate((np.linspace(1,20,20,dtype=int), np.linspace(22,50,15,dtype=int), np.linspace(55,130,numth - 35,dtype=int)))
    

seed = int(time.time())
np.random.seed(seed)
# generate igrf data
data = np.zeros((n_time,num_points))
data_igrf = np.zeros((n_time,num_points))


print('Computing covariance matrix.')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
    cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
else:
    cov = kernel(spherical2cartesian(lon,lat))
    mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

# %%
# cov = -9999 * np.ones((num_points,num_points))
# for i in range(num_points):
#     thiscov = cov_func_igrf(lat[i:], lon[i:], A, idx = 0)
#     cov[i,i:] = thiscov
# cov = np.maximum(cov,cov.T)

nbhds2, nbhds3 = {}, {}
for i in range(num_points):
    nbhds2[i] = np.where(dists[i,:] <= eps2)[0]

# %%
# needs all bins defined
def calculate_allstats(filelist, outfilename, typ, ks, robust_tolerance, dists, eps, nbhds, weighted=False, ranks=False,exp = False, alpha1 = 0.95,alpha2=0.99, var = 10):
    if not os.path.exists(base_path + outfilename):
        num_runs = len(filelist)
        all_degs, all_lls, all_ccs, all_spls = [np.zeros((numth, num_bins, num_runs)) for _ in range(4)]
        all_dens, all_llquant1, all_llquant2, all_tele1, all_tele2, all_robusttele2, all_mad = [np.zeros((numth, num_runs)) for _ in range(7)]
        all_spls = np.zeros((numth, len(splbins)-1, num_runs))
        all_shufflemad = np.zeros((numth, 100, num_runs))
        for run, nam in enumerate(filelist):
            orig_data = myload(nam)
            nam = nam.split('data_',1)[1]
            data = np.sqrt(var) * orig_data
            exp_data = np.exp(np.sqrt(var) * orig_data)
            data2 = exp_data
            # save with nam in name
            for j in range(orig_data.shape[1]):
                data[:,j] -= orig_data[:,j].mean()
                data[:,j] /= orig_data[:,j].std()
                data2[:,j] -= exp_data[:,j].mean()
                data2[:,j] /= exp_data[:,j].std()
            
            if exp:
                emp_corr = compute_empcorr(data2, similarity)
            else:
                emp_corr = compute_empcorr(data, similarity)
            for i in range(len(ks)):
                if typ == 'knn':
                    k = ks[i]
                    adj = knn_adj(np.abs(emp_corr),k, weighted = weighted)
                elif typ == 'threshold':
                    density = ks[i]
                    adj = get_adj(emp_corr, density, weighted=weighted)
                if ranks:
                    adj = rank_matrix(adj)
                    adj /= adj.max()
                deg = adj.sum(axis = 0)/(adj.shape[0]-1)
                all_dens[i,run] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))    

                #print('Avg. norm. deg of higher AR vs avg. norm. deg', deg[rd_idcs].mean(), deg.mean())
                #ar_bias[i,run] = deg[rd_idcs].mean() / deg.mean()

                print('Computing graphstats. ', time.time()-curr_time)
                curr_time = time.time()
                # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
                degnum = np.histogram(deg, bins=degbins)[0]
                all_degs[i,:,run] = degnum/num_points
                
                G = nx.from_numpy_matrix(adj)
                #idcs = np.random.permutation(adj.shape[0])[:n_estim]
                # if weighted:
                #     ccw = nx.clustering(G, weight = 'weight')
                #     ccwnum = np.histogram(list(ccw.values()),bins = degbins)[0]
                #     all_ccws[i,:,run] = ccwnum/num_points
                #     #or spl for weighted:
                #     #all_spaths = nx.shortest_path(G)
                #     #splw = np.zeros_like(all_spaths) # fix because dicts
                #     #for i in range(all_spaths.shape[0]):
                #     #    for j in range(all_spaths.shape[1]):
                #     #        splw[i,j] = len(all_spaths[i][j])-1
                # else:
                #     betwe = nx.betweenness_centrality(G, k = n_estim)
                #     betwnum = np.histogram(list(betwe.values()), bins = betwbins)[0]
                #     all_betw[i,:,run] = betwnum/num_points

                cc = nx.clustering(G) #, nodes = idcs)
                spl = dict(nx.all_pairs_shortest_path_length(G))
                thesespl = []
                for ipt in range(num_points):
                    for jpt in list(spl[ipt].keys()):
                        if jpt != ipt:
                            thesespl.append(spl[ipt][jpt])
                ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
                splnum = np.histogram(thesespl,bins = splbins)[0] / 2
                
                all_ccs[i,:,run] = ccnum/num_points
                all_spls[i,:,run] = splnum/num_points
                # accumulate over runs: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
                for ibin in range(n_dresol):
                    if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
                        plink[i,ibin] += (adj[np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1])] != 0).sum()
                    else:
                        plink[i,ibin] = np.nan
                
                print('Computing length stats. ', time.time()-curr_time)
                curr_time = time.time()
                all_tele1[i,run] = teleconn(adj, dists, 1e-8)
                all_tele2[i,run] = teleconn(adj, dists, eps)
                all_robusttele2[i,run] = robust_teleconn_nbhd(adj, dists, nbhds, tolerance = robust_tolerance)
                maxavgdeg, shufflexmaxdegs = maxavgdegbias(true_adj,nbhds)
                all_mad[i,run] = maxavgdeg
                all_shufflemad[i,:,run] = shufflexmaxdegs
                lls = dists * (adj != 0)
                if weighted:
                    lld = lldistr(dists,adj,llbins)
                    all_lls[i,:,run] = lld
                    # where does weight of all edges <= length surpass (alpha1*total_weight)
                    total_weight = adj.sum()/2
                    cum_lld = np.zeros_like(lld)
                    cum_lld[0] = lld[0]
                    for j in range(len(lld)-1):
                        cum_lld[j+1] = cum_lld[j] + lld[j+1]
                    all_llquant1[i, run] = llbins[np.where(cum_lld >= alpha1*total_weight)[0][0]]
                    all_llquant2[i, run] = llbins[np.where(cum_lld >= alpha2*total_weight)[0][0]]
                else:
                    sorted_lls = np.sort((lls)[lls != 0])
                    all_lls[i,:,run] = np.histogram(sorted_lls, bins = llbins)[0]
                    if len(sorted_lls) == 0:
                        all_llquant1[i, run] = 0
                        all_llquant2[i, run] = 0
                    else:
                        all_llquant1[i, run] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
                        all_llquant2[i, run] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]
        # after all runs
        for ibin in range(n_dresol):
            if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
                plink[:,ibin] /= (num_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())
        all_stats = [all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad]
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump(all_stats, fp)
            #np.save(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy', true_stats)
    else:
        with open(base_path + outfilename, "rb") as fp:   # Unpickling
            all_stats = pickle.load(fp)
            # true_stats = np.load(base_path + f'truestats_eps{eps2}_tol{robust_tolerance}_{len(ks)}_{weighted}_{ranks}.npy')
        all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad = all_stats
    return all_dens, all_degs, all_lls, all_ccs, all_spls, all_llquant1,all_llquant2,all_tele1,all_tele2,all_robusttele2, all_mad, all_shufflemad

# %%
# calculate edit distance for each density: how many too long links, so ll more informative..
