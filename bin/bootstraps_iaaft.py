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
num_iter = 1000
num_shuffles = 10000
ar = 0
ars = np.linspace(0,0.9,10)
ar_coeff = ar * np.ones(2)
corr = 0
sigma = np.array([[1,corr],[corr,1]])
shuffle_stats=np.zeros((6,10,num_iter))
iaaft_stats=np.zeros((6,10,num_iter))
corrs = np.zeros((10, 10*num_iter))
# %%

iaaft = np.zeros((2, num_shuffles, n_time)) # point x surrogate x time


for iar,ar in enumerate(ars):
    ar_coeff = ar * np.ones(2)
    for it in tqdm.tqdm(range(num_iter)):
        with suppress_stdout():
            #eps = np.random.multivariate_normal(np.zeros(2), sigma, size=n_time)
            eps = diag_var_process(ar_coeff, sigma, n_time)
        corrs[iar,it] = np.corrcoef(eps.T)[0,1]
        with suppress_stdout():
            for ipt in range(eps.shape[1]):
                iaaft[ipt,:,:] = surrogates(eps[:,ipt], ns = num_shuffles)
        iaaftcorrs = []
        #booteps = np.zeros_like(eps) np.random.choice(n_time,n_time)
        for isurr in range(num_shuffles):
            iaaftcorrs.append(np.corrcoef(iaaft[:,isurr,:])[0,1])
        iaaft_stats[:,iar,it] = np.mean(iaaftcorrs), np.std(iaaftcorrs), quantile(iaaftcorrs,alpha=0.5), quantile(iaaftcorrs,alpha=0.95), quantile(iaaftcorrs,alpha=0.99), quantile(iaaftcorrs,alpha=0.999)

# %%
#print(np.mean(corrs), np.std(corrs), quantile(corrs,alpha=0.5), quantile(corrs,alpha=0.95), quantile(corrs, alpha = 0.99),quantile(corrs,alpha = 0.999))
#print(shuffle_stats.mean(axis=1))
#plt.hist(corrs,bins=30)
# corrs = np.zeros((10, 10*num_iter))
# for iar,ar in enumerate(ars):
#     ar_coeff = ar * np.ones(2)
#     for it in tqdm.tqdm(range(10*num_iter)):
#         with suppress_stdout():
#             #eps = np.random.multivariate_normal(np.zeros(2), sigma, size=n_time)
#             eps = diag_var_process(ar_coeff, sigma, n_time)
#         corrs[iar,it] = np.corrcoef(eps.T)[0,1]

#mysave(base_path, f'resampling_arbias_iaaft_{n_time}_{num_iter}_{num_shuffles}.txt',[iaaft_stats])

#sys.exit(0)


# %%
# how good is resampling on only one realization of n=100? does it make a difference that they were dependent? yes, for large ar
# which density would we then get in our networks? how good is the resulting fdr?
import time
curr_time = time.time()
base_path = '../../climnet_output/'
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
#%%
ys = kernel(spherical2cartesian(180/np.pi * xs, np.zeros_like(xs)))[0,:]
np.where(ys>quantile(corrs[iar,:],alpha=0.95))[0][-1]
# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
n_time = 100
num_iter = 1000
num_shuffles = 10000
corrs,shuffle_stats,boot_stats, confboot_stats = myload(base_path + f'resampling_arbias_{n_time}_{num_iter}_{num_shuffles}.txt')
iaaft_stats = myload(find(f'resampling_arbias_iaaft_{n_time}_{num_iter}_{num_shuffles}.txt',base_path)[0])[0]
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
plt.savefig(base_path+f'resampling3_arbias_0.95_iaaft_{n_time}_{num_iter}_{num_shuffles}.pdf')

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

# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
n_time = 100
num_iter = 1000
num_shuffles = 10000
corrs,shuffle_stats,boot_stats, confboot_stats = myload(base_path + f'resampling_arbias_{n_time}_{num_iter}_{num_shuffles}.txt')
iaaft_stats = myload(find(f'resampling_arbias_iaaft_{n_time}_{num_iter}_{num_shuffles}.txt',base_path)[0])[0]
fig,axs = plt.subplots(1,3,figsize=(3*onefigsize[0],1.2*onefigsize[1]))
axs[0].plot(ars, [quantile(corrs[iar,:],alpha=0.95) for iar in range(len(ars))], label = 'true',color = 'black')
axs[0].plot(ars, shuffle_stats[3,:,:].mean(axis=1), label = 'shuffled')
#axs[0].fill_between(ars, shuffle_stats[3,:,:].mean(axis = 1) - 2 *shuffle_stats[3,:,:].std(axis = 1), shuffle_stats[3,:,:].mean(axis = 1) + 2 *shuffle_stats[3,:,:].std(axis = 1), alpha = 0.4)
axs[0].fill_between(ars, [quantile(shuffle_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(shuffle_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#axs[0].plot(ars, boot_stats[3,:,:].mean(axis=1), label = 'naively bootstrapped')
#axs[0].fill_between(ars, boot_stats[3,:,:].mean(axis = 1) - 2 *boot_stats[3,:,:].std(axis = 1), boot_stats[3,:,:].mean(axis = 1) + 2 *boot_stats[3,:,:].std(axis = 1), alpha = 0.4)
#axs[0].fill_between(ars, [quantile(boot_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(boot_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
axs[0].plot(ars, iaaft_stats[3,:,:].mean(axis=1), label = 'IAAFT')
#axs[0].fill_between(ars, iaaft_stats[3,:,:].mean(axis = 1) - 2 *iaaft_stats[3,:,:].std(axis = 1), iaaft_stats[3,:,:].mean(axis = 1) + 2 *iaaft_stats[3,:,:].std(axis = 1), alpha = 0.4)
axs[0].fill_between(ars, [quantile(iaaft_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(iaaft_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#axs[0].plot(ars, confboot_stats[0,:,:].mean(axis=1), label = 'advanced bootstrap')
axs[0].set_xlabel('Autocorrelation')
axs[0].set_ylabel('0.95-quantile')
axs[0].legend()

all_dens,all_fdr,all_fdrz,arbiash,arbiasl,arbiashz,arbiaslz = myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
filename = find(f'allarbiasquants_*ar0.2_0.7_time*',base_path+'signif/')[-1]
all_dens_shuff, all_fdr_shuff, all_fdrthres_shuff, arbiash_shuff, arbiasl_shuff,tarbiash_shuff,tarbiasl_shuff,ar_coeff, alphas = myload(base_path+'signif/'+ f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1])
p1,=axs[1].plot(all_dens[1:], arbiash[1:], color = adjust_lightness('tab:grey', amount=1.5), label='t,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p2,=axs[1].plot(all_dens[1:], arbiasl[1:], color = adjust_lightness('tab:grey', amount=0.5), label='t,low AR')#, color = adjust_lightness('tab:blue', amount=0.5))
p5,=axs[1].plot(all_dens[1:], arbiashz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='iaaft,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p6,=axs[1].plot(all_dens[1:], arbiaslz[1:], color = adjust_lightness('tab:orange', amount=0.5), label='iaaft,low AR')
p3,=axs[1].plot(all_dens_shuff[1:], arbiash_shuff[1:], color = adjust_lightness('tab:blue', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p4,=axs[1].plot(all_dens_shuff[1:], arbiasl_shuff[1:], color = adjust_lightness('tab:blue', amount=0.5), label='s,low AR')
# p7,=axs[1].plot(all_densb[1:], ensarbiash2[1:], color = adjust_lightness('tab:red', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
# p8,=axs[1].plot(all_densb[1:], ensarbiasl2[1:], color = adjust_lightness('tab:red', amount=0.5), label='s,low AR')
axs[1].set_xlabel('Density')
axs[1].set_ylabel('Avg. norm. degree')
#axs[1].set_xlim(-0.005,0.2)
l = axs[1].legend([(p1, p2),(p3,p4),(p5,p6)], ['threshold','shuffles','IAAFT'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)


filename = find(f'allarbiasquants_*ar0.2_time*',base_path+'signif/')[-1]
all_dens_low, all_fdr_low, all_fdrthres_low, alphas = myload(base_path+'signif/'+ f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1])
all_dens,all_fdr_lowiaaft,all_fdrz_lowiaaft = myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_time100.txt')
axs[2].set_ylabel('FDR')
axs[2].set_xlabel('Density')
p12,=axs[2].plot(all_dens[1:],all_fdrz_lowiaaft[1:], color = adjust_lightness('tab:orange', amount=0.7), label='signif, low AR')
#p1,=axs[2].plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='signif, high AR')
p2,=axs[2].plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.3), label='signif, mixed AR')

#p2,=axs[2].plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:orange', amount=1), label='signif, mixed AR')
#p3,=axs[2].plot(all_densh[1:],all_fdrhthres[1:], color = adjust_lightness('tab:blue', amount=1.5),label = 'thres, high AR')
p34,=axs[2].plot(all_dens[1:],all_fdr_lowiaaft[1:], color = adjust_lightness('tab:grey', amount=0.7),label = 'thres, low AR')
p4,=axs[2].plot(all_dens[1:],all_fdr[1:], color = adjust_lightness('tab:grey', amount=1.3), label='thres, mixed AR')

p56,=axs[2].plot(all_dens_low[1:],all_fdr_low[1:], color = adjust_lightness('tab:blue', amount=0.7),label = 'thres, low AR')
p6,=axs[2].plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:blue', amount=1.3), label='thres, mixed AR')
#p5,=axs[2].plot(all_densensh[1:],all_fdrens1h[1:], color = adjust_lightness('tab:green', amount=1.5),label = 'ens1, high AR')
#p6,=axs[2].plot(all_densens[1:],all_fdrens1[1:], color = adjust_lightness('tab:green', amount=1),label = 'ens1, mixed AR')
#p7,=axs[2].plot(all_densh[1:],all_fdrens2h[1:], color = adjust_lightness('tab:red', amount=1.5), label='ens2, high AR')
#p8,=axs[2].plot(all_densb[1:],all_fdrens2[1:], color = adjust_lightness('tab:red', amount=1), label='ens2, mixed AR')
l = axs[2].legend([(p4,p34),(p6,p56),(p2,p12)], ['threshold','shuffles','IAAFT'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False) ,(p5,p6),(p7,p8)] ,'Stab. selection','Bootstr. mean'
axs[2].set_xlim(0.002,0.1)
axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path + f'joint_iaaft_plot.pdf')

# %%
all_dens_shuff,all_fdr_quant,all_fdr_thres, all_fdr_z,arbiash_thres,arbiash_quant,arbiash_z,arbiasl_thres,arbiasl_quant,arbiasl_z=myload(base_path+'signif/all_iaaft_zvsq_stats_0.2_0.7.txt')
#long_dens,long_fdr,long_fdrz,longarbiash,longarbiasl,longarbiashz,longarbiaslz=myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')


from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
n_time = 100
num_iter = 1000
num_shuffles = 10000
corrs,shuffle_stats,boot_stats, confboot_stats = myload(base_path + f'resampling_arbias_{n_time}_{num_iter}_{num_shuffles}.txt')
iaaft_stats = myload(find(f'resampling_arbias_iaaft_{n_time}_{num_iter}_{num_shuffles}.txt',base_path)[0])[0]
fig,axs = plt.subplots(1,3,figsize=(3*onefigsize[0],1.2*onefigsize[1]))
axs[0].plot(ars, [quantile(corrs[iar,:],alpha=0.95) for iar in range(len(ars))], label = 'true',color = 'black')
axs[0].plot(ars, shuffle_stats[3,:,:].mean(axis=1), label = 'random')
#axs[0].fill_between(ars, shuffle_stats[3,:,:].mean(axis = 1) - 2 *shuffle_stats[3,:,:].std(axis = 1), shuffle_stats[3,:,:].mean(axis = 1) + 2 *shuffle_stats[3,:,:].std(axis = 1), alpha = 0.4)
axs[0].fill_between(ars, [quantile(shuffle_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(shuffle_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#axs[0].plot(ars, boot_stats[3,:,:].mean(axis=1), label = 'naively bootstrapped')
#axs[0].fill_between(ars, boot_stats[3,:,:].mean(axis = 1) - 2 *boot_stats[3,:,:].std(axis = 1), boot_stats[3,:,:].mean(axis = 1) + 2 *boot_stats[3,:,:].std(axis = 1), alpha = 0.4)
#axs[0].fill_between(ars, [quantile(boot_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(boot_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
axs[0].plot(ars, iaaft_stats[3,:,:].mean(axis=1), label = 'IAAFT')
#axs[0].fill_between(ars, iaaft_stats[3,:,:].mean(axis = 1) - 2 *iaaft_stats[3,:,:].std(axis = 1), iaaft_stats[3,:,:].mean(axis = 1) + 2 *iaaft_stats[3,:,:].std(axis = 1), alpha = 0.4)
axs[0].fill_between(ars, [quantile(iaaft_stats[3,iar,:],alpha=0.025) for iar in range(len(ars))], [quantile(iaaft_stats[3,iar,:],alpha=0.975) for iar in range(len(ars))], alpha = 0.4)
#axs[0].plot(ars, confboot_stats[0,:,:].mean(axis=1), label = 'advanced bootstrap')
axs[0].set_xlabel('Autocorrelation')
axs[0].set_ylabel('0.95-quantile')
axs[0].legend()

all_dens,all_fdr,all_fdrz,arbiash,arbiasl,arbiashz,arbiaslz = myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
all_densshuff,all_fdrshuff,all_fdrzshuff,arbiashshuff,arbiaslshuff,arbiashzshuff,arbiaslzshuff = myload(base_path+'signif/'+f'all_shuffstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
all_denssubsample,all_fdrsubsample,arbiashsubsample,arbiaslsubsample = myload(base_path+'signif/'+f'all_subsamplestats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')


#filename = find(f'allarbiasquants_*ar0.2_0.7_time*',base_path+'signif/')[-1]
#all_dens_shuff, all_fdr_shuff, all_fdrthres_shuff, arbiash_shuff, arbiasl_shuff,tarbiash_shuff,tarbiasl_shuff,ar_coeff, alphas = myload(base_path+'signif/'+ f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1])
p1,=axs[1].plot(all_dens[1:], arbiash[1:]/all_dens[1:], color = adjust_lightness('tab:blue', amount=1.5), label='t,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p2,=axs[1].plot(all_dens[1:], arbiasl[1:]/all_dens[1:], color = adjust_lightness('tab:blue', amount=0.5), label='t,low AR')#, color = adjust_lightness('tab:blue', amount=0.5))
p5,=axs[1].plot(all_dens[1:], arbiashz[1:]/all_dens[1:], color = adjust_lightness('tab:orange', amount=1.5), label='iaaft,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p6,=axs[1].plot(all_dens[1:], arbiaslz[1:]/all_dens[1:], color = adjust_lightness('tab:orange', amount=0.5), label='iaaft,low AR')
p3,=axs[1].plot(all_dens_shuff[1:], arbiash_quant[1:]/all_dens_shuff[1:],'x', color = adjust_lightness('tab:green', amount=1.5), label='q,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p4,=axs[1].plot(all_dens_shuff[1:], arbiasl_quant[1:]/all_dens_shuff[1:],'x', color = adjust_lightness('tab:green', amount=0.5), label='q,low AR')
# p7,=axs[1].plot(all_densb[1:], ensarbiash2[1:], color = adjust_lightness('tab:red', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
# p8,=axs[1].plot(all_densb[1:], ensarbiasl2[1:], color = adjust_lightness('tab:red', amount=0.5), label='s,low AR')
axs[1].set_xlabel('Density')
axs[1].set_ylabel('Avg. norm. degree')
#axs[1].set_xlim(-0.005,0.2)
l = axs[1].legend([(p1, p2),(p5,p6),(p3,p4)], ['thresh/random','IAAFT zscore','IAAFT quant'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)


filename = find(f'allarbiasquants_*ar0.2_time*',base_path+'signif/')[-1]
#all_dens_low, all_fdr_low, all_fdrthres_low, alphas = myload(base_path+'signif/'+ f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1])
#all_dens,all_fdr_lowiaaft,all_fdrz_lowiaaft = myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_time100.txt')
axs[2].set_ylabel('FDR')
axs[2].set_xlabel('Density')
#p12,=axs[2].plot(all_dens[1:],all_fdrz_lowiaaft[1:], color = adjust_lightness('tab:orange', amount=0.7), label='signif, low AR')
#p1,=axs[2].plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='signif, high AR')

axs[2].plot(all_dens[1:],all_fdr[1:], color = adjust_lightness('tab:blue', amount=1), label='thresh/random')
axs[2].plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1), label='IAAFT zscore')
#p2,=axs[2].plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:orange', amount=1), label='signif, mixed AR')
#p3,=axs[2].plot(all_densh[1:],all_fdrhthres[1:], color = adjust_lightness('tab:blue', amount=1.5),label = 'thres, high AR')
#p34,=axs[2].plot(all_dens[1:],all_fdr_lowiaaft[1:], color = adjust_lightness('tab:grey', amount=0.7),label = 'thres, low AR')
axs[2].plot(all_dens_shuff[1:],all_fdr_quant[1:],'x', color = adjust_lightness('tab:green', amount=1),label = 'IAAFT quant')

#p56,=axs[2].plot(all_dens_low[1:],all_fdr_low[1:], color = adjust_lightness('tab:blue', amount=0.7),label = 'thres, low AR')
#axs[2].plot(all_denssubsample[1:],all_fdrsubsample[1:], color = adjust_lightness('tab:blue', amount=1), label='subsampling')
#p5,=axs[2].plot(all_densensh[1:],all_fdrens1h[1:], color = adjust_lightness('tab:green', amount=1.5),label = 'ens1, high AR')
#p6,=axs[2].plot(all_densens[1:],all_fdrens1[1:], color = adjust_lightness('tab:green', amount=1),label = 'ens1, mixed AR')
#p7,=axs[2].plot(all_densh[1:],all_fdrens2h[1:], color = adjust_lightness('tab:red', amount=1.5), label='ens2, high AR')
#p8,=axs[2].plot(all_densb[1:],all_fdrens2[1:], color = adjust_lightness('tab:red', amount=1), label='ens2, mixed AR')
#l = axs[2].legend([(p4,p34),(p6,p56),(p2,p12)], ['threshold','random','IAAFT'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False) ,(p5,p6),(p7,p8)] ,'Stab. selection','Bootstr. mean'
axs[2].legend()
axs[2].set_xlim(0.002,0.1)
axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path + f'joint_iaaft_plot_relativedegree_new_inclquantcross.png')






# %%
# alphas = np.linspace(0.5,1,50)
# y = [quantile(corrs,alpha=alpha) for alpha in alphas]
# fig,ax = plt.subplots()
# ax.plot(alphas,y,label='independent')
# #ax.scatter(quantile(corrs,alpha=0.999),0,s=10,marker='+',color = 'tab:blue')
# ax.scatter(0.5,shuffle_stats[2,:].mean(),marker='x',label='resampled',color = 'tab:orange')
# ax.scatter(0.95,shuffle_stats[3,:].mean(),marker='x', color = 'tab:orange')
# ax.scatter(0.99,shuffle_stats[4,:].mean(),marker='x',color = 'tab:orange')
# ax.scatter(0.999,shuffle_stats[5,:].mean(),marker='x',color = 'tab:orange')
# ax.set_xlabel('Quantile')
# ax.set_ylabel('Emp corr')
# ax.legend()

