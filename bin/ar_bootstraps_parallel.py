# %%
from matplotlib import rcParams
import networkx as nx
import os
import numpy as np
import scipy.interpolate as interp
from scipy import stats, ndimage
import scipy
import tueplots
import xarray as xr
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import revised_mi
#from multiprocessing import Pool
import time
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
#plt.style.use('bmh')
irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', irun)
curr_time = time.time()

def autocorr(X, t = 1):
    '''
    Computes autocorr at lag 1 between columns of X.
    '''
    d = X.shape[1]
    return np.corrcoef(X[:-t,:], X[t:,:],rowvar = False)[:d,d:]

# def eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx, n_pre = 100):
#     dim = cov.shape[0]
#     data = np.zeros((n_time+n_pre,dim))
#     eigA = np.zeros_like(eigvecs)
#     eigA[pc_idx,pc_idx] = ar_pc
#     A = eigvecs @ eigA @ eigvecs.T
#     eigvals_eps = eigvals
#     eigvals_eps[pc_idx] = eigvals[pc_idx] * (1-ar_pc ** 2)
#     sigma = eigvecs @ np.diag(eigvals_eps) @ eigvecs.T
#     eps = np.random.multivariate_normal(np.zeros(dim), sigma, size=n_time+n_pre)
#     data[0,:] = np.random.multivariate_normal(np.zeros(dim), eigvecs @ np.diag(eigvals) @ eigvecs.T)
#     for i in range(n_pre+n_time-1):
#         data[i+1,:] = A @ data[i,:] + eps[i+1,:]
#     return data[n_pre:,:]

num_runs = 1
num_tasks = 30
longtime = 10000
#number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])


# Network parameters
n_lat = 18 * 4 # makes resolution of 180 / n_lat degrees
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = False
ranks = False
corr_method='pearson' # 'spearman', 'MI', 'HSIC', 'ES'
pcar = False
denslist = [0.001,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
ks = [6, 60, 300, 600,1200]
alphas = [0.5,0.9,0.95,0.99,0.999]
#ks = [5,  10, 65, 125, 250]
if len(denslist) != len(ks) and typ == 'threshold':
    raise RuntimeError('Denslist needs to have same length as ks.')

robust_tolerance = 0.5
# data parameters
distrib = 'igrf'
n_time = 100
nu = 1.5
len_scale = 0.2


exec(open("grid_helper.py").read())

# %%
print('Computing covariance matrix.')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

ac = 0.2
#ar = ac
#ac2 = 0.7
ar_coeff = ac * np.ones(num_points)
#rd_idcs = np.random.permutation(np.arange(num_points))[:num_points // 2]
#ar_coeff[rd_idcs] = ac2 * np.ones(len(rd_idcs))

#datacov = true_cov_diag_var(ar_coeff,cov)


# %%
def compute_edges(data, alphas, mtx_idcs,mtx_idcsj, num_shuffles = 1000):
    if len(mtx_idcs) != len(mtx_idcsj):
        raise ValueError(f'Incompatible index lengths: {len(mtx_idcs)} != {len(mtx_idcsj)}')
    quants = {}
    for idx in range(len(mtx_idcs)):
        i,j = mtx_idcs[idx], mtx_idcsj[idx]
        #print(i,j)
        eps =  data[:,[i,j]]
        shufflecorrs = []
        for _ in range(num_shuffles):
            eps[:,0] = eps[np.random.permutation(data.shape[0]),0]
            shufflecorrs.append(np.corrcoef(eps.T)[0,1])
        quants[(i,j)] =  [quantile(shufflecorrs,alpha = alpha) for alpha in alphas]
        #print( [quantile(shufflecorrs,alpha = alpha) for alpha in alphas])
    return quants

def compute_edges_iaaft(data,iaaft, alphas, mtx_idcs,mtx_idcsj):
    if len(mtx_idcs) != len(mtx_idcsj):
        raise ValueError(f'Incompatible index lengths: {len(mtx_idcs)} != {len(mtx_idcsj)}')
    num_shuffles = iaaft.shape[1]
    quants = {}
    mean_std = {}
    for idx in range(len(mtx_idcs)):
        i,j = mtx_idcs[idx], mtx_idcsj[idx]
        #print(i,j)
        #eps =  data[:,[i,j]]
        shufflecorrs = []
        for isurr in range(num_shuffles):
            shufflecorrs.append(np.corrcoef(iaaft[[i,j],isurr,:])[0,1])
        quants[(i,j)] =  [quantile(shufflecorrs,alpha = alpha) for alpha in alphas]
        mean_std[(i,j)] = [np.mean(shufflecorrs), np.std(shufflecorrs)]
        #print( [quantile(shufflecorrs,alpha = alpha) for alpha in alphas])
    return quants, mean_std
# %%
n_dresol = 9
truecorrbins = np.linspace(0,1,n_dresol+1)
plink = np.zeros((len(denslist),n_dresol))
plinkh = np.zeros((len(denslist),n_dresol))
plinkl = np.zeros((len(denslist),n_dresol))
run = 0
num_runs = 1
curr_time=time.time()
datacov = true_cov_diag_var(ar_coeff,cov)
print(f'Starting {run}th run. ', time.time()-curr_time)
empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ac}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
if empdatas == []:
    seed = int(time.time())
    np.random.seed(seed)
    orig_data = diag_var_process(ar_coeff, cov, n_time)
    mysave(base_path+'empdata/',f'arbiasdata_nu{nu}_len{len_scale}_ar{ac}_{grid_type}{n_lat}_time{n_time}_{seed}.txt',orig_data)    
else:
    orig_data = myload(empdatas[0])
data = 1 * orig_data
# save with nam in name
for j in range(len(lat)):
    data[:,j] -= orig_data[:,j].mean()
    data[:,j] /= orig_data[:,j].std()

print('Computing similarity matrix. ', time.time()-curr_time)
curr_time = time.time()
#emp_corr = compute_empcorr(data, similarity=corr_method)
#adj = np.zeros((data.shape[1],data.shape[1]))
corr = np.corrcoef(data.T)

# split uppercorr in equal parts and compute in parallel
mtx_idcs = np.triu_indices_from(corr,k=1)
num_edges = len(mtx_idcs[0])
args = []
for i in range(num_tasks):
    args.append((data,alphas, mtx_idcs[0][int(i*num_edges/num_tasks):int((i+1)*num_edges/num_tasks)],mtx_idcs[1][int(i*num_edges/num_tasks):int((i+1)*num_edges/num_tasks)]))

# %%
num_shuffles = 1000
iaaft = np.zeros((data.shape[1], num_shuffles, data.shape[0]))
with suppress_stdout():
    for ipt in range(data.shape[1]):
        iaaft[ipt,:,:] = surrogates(data[:,ipt], ns = num_shuffles)

# %%
quants, mean_std = compute_edges_iaaft(data, iaaft, alphas, mtx_idcs[0][int(irun*num_edges/num_tasks):int((irun+1)*num_edges/num_tasks)],mtx_idcs[1][int(irun*num_edges/num_tasks):int((irun+1)*num_edges/num_tasks)])
mysave(base_path+'signif/',f'arbiasquants_iaaft_part{irun}_nu{nu}_len{len_scale}_ar{ac}_time{n_time}.txt',[quants, mean_std,data,alphas])
# with Pool(os.cpu_count()) as pool:
#     dict_list = pool.starmap(compute_edges, args)

# allquants = dict_list[0]
# for i in range(len(dict_list)-1):
#     allquants.update(dict_list[i+1])

# mysave(base_path,f'arbias_quants_signif_nu{nu}_len{len_scale}_{ac}_{ac2}_time{n_time}.txt',[allquants,ar_coeff,alphas])

# %% 
def ij_from_idx(idx,m,n=None):
    if n is None:
        n = m
    # idx = i * n + j
    j = idx % n
    i = int((idx - j)/n)
    return i,j

# %%
#quants,ar_coeff,alphas = myload(find('arbias_quants_part0_*',base_path+'signif/')[0])
# %%
len(quants.keys())