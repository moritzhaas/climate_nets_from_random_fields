# %%
# calculate the 0.95,0.99,0.999 quantiles of empirical correlation between 2 independent or autocorrelated normals with n=100
import numpy as np
import matplotlib.pyplot as plt
from climnet.myutils import *
from contextlib import contextmanager
#from copy import deepcopy
import tqdm, sys
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})

base_path = '../../climnet_output/'

from scipy.stats import bootstrap
def mycorr(x,y):
    return np.corrcoef(np.array([x,y]))[0,1]

def myboot(eps, alpha=0.9, num_shuffles=1000):
    return bootstrap((eps[:,0],eps[:,1]),mycorr,paired=True,vectorized=False,confidence_level=alpha,n_resamples=num_shuffles).confidence_interval[1]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

n_time = 100
num_iter = 100
num_shuffles = 10000
ar = 0
ars = np.linspace(0,0.9,10)
ar_coeff = ar * np.ones(2)
corr = 0
sigma = np.array([[1,corr],[corr,1]])

# %%
# how good is resampling on only one realization of n=100? does it make a difference that they were dependent? yes, for large ar
# which density would we then get in our networks? how good is the resulting fdr?
import time
curr_time = time.time()
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='pearson' #dont change!
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True
alphas = [0.9,0.95,0.99,0.995,0.999]

ar = 0
ar2 = None
var = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5
len_scale = 0.1

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
empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
#for w in [True,False]:
    # if typ == 'knn' and not w:
    #     continue
outfilename = f'signifiaaft{num_shuffles}_{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
for irun in range(num_runs):
    all_dens = np.zeros((numth, num_runs))
    sig_fdr = np.zeros((numth, num_runs))
    thr_fdr = np.zeros((numth, num_runs))

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
    for i, alpha in enumerate(alphas):
        sig_adj = get_significance_adj(data,alpha=alpha, num_shuffles=num_shuffles,weighted = False, use_iaaft = True)
        density = sig_adj.sum()/ (sig_adj.shape[0]*(sig_adj.shape[0]-1))
        all_dens[i,irun] = density
        adj = get_adj(emp_corr, density, weighted=False)
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        sig_fdr[i,irun] = sig_adj[np.logical_and(sig_adj != 0, dists > maxdist)].sum() / sig_adj[sig_adj!=0].sum()#np.logical_and(sig_adj != 0, dists > maxdist).sum() / (sig_adj != 0).sum()
        thr_fdr[i,irun] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()#np.logical_and(adj != 0, dists > maxdist).sum() / (adj != 0).sum()
all_stats = [all_dens,sig_fdr,thr_fdr] # all_tele3, all_robusttele3,
mysave(base_path,outfilename,all_stats)

# %%
# which density would we get in our ground truth networks if we choose the significance threshold from above?
from sklearn.gaussian_process.kernels import Matern
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
n_lat = 72
len_scale = 0.1
nu=0.5
num_points = gridstep_to_numpoints(180/n_lat)
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

xs = np.linspace(0,np.pi,200)
ys = kernel(spherical2cartesian(180/np.pi * xs, np.zeros_like(xs)))[0,:]
radii = [xs[np.where(ys>quantile(corrs[iar,:],alpha=0.95))[0][-1]] for iar in range(len(ars))] # 
true_densities = (1-np.cos(radii))/2
print(nu, len_scale,': ', true_densities[[0,-1]])

# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
n_time = 100
num_iter = 1000
num_shuffles = 10000
corrs,shuffle_stats,boot_stats, confboot_stats = myload(base_path + f'resampling_arbias_{n_time}_{num_iter}_{num_shuffles}.txt')
iaaft_stats = myload(find(f'resampling_arbias_iaaft_{n_time}_*.txt',base_path)[0])[0]
fig,ax = plt.subplots()
ax.plot(ars, [quantile(corrs[iar,:],alpha=0.95) for iar in range(len(ars))], label = 'true',color = 'black')
ax.plot(ars, shuffle_stats[3,:,:].mean(axis=1), label = 'shuffled')
#ax.fill_between(ars, shuffle_stats[3,:,:].mean(axis = 1) - 2 *shuffle_stats[3,:,:].std(axis = 1), shuffle_stats[3,:,:].mean(axis = 1) + 2 *shuffle_stats[3,:,:].std(axis = 1), alpha = 0.4)
ax.fill_between(ars, [quantile(shuffle_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(shuffle_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#ax.plot(ars, boot_stats[3,:,:].mean(axis=1), label = 'naively bootstrapped')
#ax.fill_between(ars, boot_stats[3,:,:].mean(axis = 1) - 2 *boot_stats[3,:,:].std(axis = 1), boot_stats[3,:,:].mean(axis = 1) + 2 *boot_stats[3,:,:].std(axis = 1), alpha = 0.4)
#ax.fill_between(ars, [quantile(boot_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(boot_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
ax.plot(ars, iaaft_stats[3,:,:].mean(axis=1), label = 'IAAFT')
#ax.fill_between(ars, iaaft_stats[3,:,:].mean(axis = 1) - 2 *iaaft_stats[3,:,:].std(axis = 1), iaaft_stats[3,:,:].mean(axis = 1) + 2 *iaaft_stats[3,:,:].std(axis = 1), alpha = 0.4)
ax.fill_between(ars, [quantile(iaaft_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(iaaft_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#ax.plot(ars, confboot_stats[0,:,:].mean(axis=1), label = 'advanced bootstrap')
ax.set_xlabel('Autocorrelation')
ax.set_ylabel('0.95-quantile')
ax.legend()
plt.savefig(base_path+f'resampling3_arbias_0.95_iaaft_{num_iter}_{num_shuffles}.pdf')

# %%
fig,ax = plt.subplots()
ax.plot(ars, [quantile(corrs[iar,:],alpha=0.99) for iar in range(len(ars))], label = 'true',color = 'black')
ax.plot(ars, shuffle_stats[4,:,:].mean(axis=1), label = 'shuffled')
ax.fill_between(ars, [quantile(shuffle_stats[4,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(shuffle_stats[4,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
ax.plot(ars, confboot_stats[1,:,:].mean(axis=1), label = 'bootstrapped')
ax.fill_between(ars, [quantile(confboot_stats[1,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(confboot_stats[1,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)

#ax.plot(ars, confboot_stats[1,:,:].mean(axis=1), label = 'advanced bootstrap')
ax.set_xlabel('Autocorrelation')
ax.set_ylabel('0.99-quantile')
ax.legend()
plt.savefig(base_path+f'resampling3_arbias_0.99_{num_iter}_{num_shuffles}.pdf')