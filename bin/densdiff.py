# %%
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
#from numpy.lib.type_check import nan_to_num
from scipy import signal, stats
# import cartopy as ctp
import xarray as xr
from climnet.dataset_new import AnomalyDataset
from climnet.grid import regular_lon_lat, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import *
import time
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()
curr_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})

print('HOME is: ', os.getenv("HOME"))
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='pearson'
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True

var_name = 't2m' # ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr'] #, 'sp'
var_names = ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr', 'sp']

denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
# ks = [6, 60, 300, 600,1200]

#filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'


#grid_helper and calc_true
num_runs = 2
n_time = 100
nu = 0.5
len_scale = 0.1
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99
robust_tolerance = 0.2


# %%
n_boot = 10
weighted = False
denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
seed = 1204
timeparts = 10
density = 0.01
# Grid
grid_step = 180 / n_lat
alpha=0.05
rm_outliers = True
#num_cpus = 48
#lon_range = [-110, 40]
#lat_range = [20, 85]
save = True
# time_range = ['1980-01-01', '2019-12-31']
time_range = None
tm = None  # 'week'
norm = False
#detrend = True
absolute = False

PATH = os.path.dirname(os.path.abspath(__file__))
print('You are here: ', PATH)


base_path = '../../climnet_output/'
base_path = base_path + 'real/'
if not os.path.exists(base_path):
    os.mkdir(base_path)

lenbins = np.linspace(0,np.pi,100)

# %%
for var_name in var_names:
    if (os.getenv("HOME") == '/Users/moritz') or (os.getenv("HOME") is None):    
        dataset_nc = base_path + f"era5_singlelevel_monthly_temp2m_1979-2020.nc"
        #dataset_nc = '/Volumes/backups2/2m_temperature_sfc_1979_2020.nc'
    else:# ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr'] #, 'sp'
        if var_name == 't2m':
            dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
            vname = var_name
        elif var_name == 't2mdaily':
            dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/2m_temperature_sfc_1979_2020.nc"
            vname = 't2m'
        elif var_name == 'z250':
            dataset_nc = f"/mnt/qb/goswami/data/era5/multi_pressure_level_monthly/geopotential/250/geopotential_250_1979_2020.nc"
            vname = 'z'
        elif var_name == 'z500':
            dataset_nc = f"/mnt/qb/goswami/data/era5/multi_pressure_level_monthly/geopotential/500/geopotential_500_1979_2020.nc"
            vname = 'z'
        elif var_name == 'z850':
            dataset_nc = f"/mnt/qb/goswami/data/era5/multi_pressure_level_monthly/geopotential/850/geopotential_850_1979_2020.nc"
            vname = 'z'
        elif var_name == 'pr':
            dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
            vname = var_name
        elif var_name == 'sp':
            dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/surface_pressure/surface_pressure_sfc_1979_2020.nc"
            vname = var_name

    mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
    ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False)
    data = ds.ds['anomalies'].data
    print(data.shape)
    empcorr = np.corrcoef(data.T)
    max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
    emp_corr = compute_empcorr(data, 'pearson')
    adj = get_adj(emp_corr, density, weighted=False)
    vmin, vmax = 0, adj.sum(axis=1).max()


    # %%
    # idens x iboot x ibin
    # calc, save, plot densdiff :)
    # bootstrap data in time
    denslistplus = deepcopy(denslist)
    denslistplus.append(density)
    
    print(f'Computing densdiff for {var_name}.')
    dens_diff, lldistrib,difflldistrib = [],[],[]
    for dens in denslistplus:
        thisdens_diff, thislldistrib, thisdifflldistrib = [],[],[]
        
        for iboot in range(n_boot):
            adj1 = get_adj(get_bootcorr(data, seed+2*iboot),dens= dens,weighted=weighted)
            adj2 = get_adj(get_bootcorr(data, seed+2*iboot+1),dens= dens,weighted=weighted)
            diff = np.where(np.abs(adj1 - adj2)!=0)
            thisdens_diff.append(len(diff[0]) / (dens * adj1.shape[0] * (adj1.shape[0]-1)))
            thislldistrib.append(np.histogram(dists[np.where(adj1!=0)],lenbins)[0])
            thislldistrib.append(np.histogram(dists[np.where(adj2!=0)],lenbins)[0])
            thisdifflldistrib.append(np.histogram(dists[diff],lenbins)[0])
        lldistrib.append(thislldistrib)
        difflldistrib.append(thisdifflldistrib)
        dens_diff.append(thisdens_diff)
    lldistrib = np.array(lldistrib)
    difflldistrib = np.array(difflldistrib)
    dens_diff = np.array(dens_diff)
    mysave(base_path, f'densdiff_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt', dens_diff)
    mysave(base_path, f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt', [lenbins, lldistrib, difflldistrib])

# %%
densdiff_dict = {}
diffmean, diffstd, diffmin, diffmax = {},{},{},{}
for var_name in var_names:
    print(f'Loading densdiff for {var_name}.')
    densdiff_dict[var_name] = myload(base_path + f'densdiff_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')
    
    diffmean[var_name], diffstd[var_name], diffmin[var_name], diffmax[var_name] = [np.zeros(len(denslist)) for _ in range(4)]
    for idens, dens in enumerate(denslist):
        diffmean[var_name][idens] = np.mean(densdiff_dict[var_name][idens])
        diffstd[var_name][idens] = np.std(densdiff_dict[var_name][idens])
        diffmin[var_name][idens] = quantile(densdiff_dict[var_name][idens],alpha=0.025)
        diffmax[var_name][idens] = quantile(densdiff_dict[var_name][idens],alpha=0.975)

# %%
nus = [0.5,1.5]
len_scales = [0.1,0.2]

densdiffsim = myload('../../climnet_output/real/' + f'densdiff_sim_{corr_method}_w{weighted}_{n_boot}.txt')
for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        var_name = f'nu={nu}, len={len_scale}'
        diffmean[var_name], diffstd[var_name], diffmin[var_name], diffmax[var_name] = [np.zeros(len(denslist)) for _ in range(4)]
        for idens, dens in enumerate(denslist):
            diffmean[var_name][idens] = np.mean(densdiffsim[inu,ilen,idens,:])
            diffstd[var_name][idens] = np.std(densdiffsim[inu,ilen,idens,:])
            diffmin[var_name][idens] = quantile(densdiffsim[inu,ilen,idens,:],alpha=0.025)
            diffmax[var_name][idens] = quantile(densdiffsim[inu,ilen,idens,:],alpha=0.975)

#dens_diff[inu,ilen,idens,irun]
# %%
# larger relative difference because independent redraws instead of bootstrap of the same AR-dataset..
import string
def enumerate_subplots(axs, pos_x=-0.08, pos_y=1.05, fontsize=16):
    """Adds letters to subplots of a figure.
    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.
    Returns:
        axs (list): List of plt.axes.
    """
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            pos_x[n],
            pos_y[n],
            f"{string.ascii_lowercase[n]}.",
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
        )
    plt.tight_layout()
    return axs
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
var_names2 = ['t2m', 'pr', 'sp', 'z850'] # ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr', 'sp']
# for inu,nu in enumerate(nus):
#     for ilen,len_scale in enumerate(len_scales):
#         var_names2.append(f'nu={nu}, len={len_scale}')
#         print(diffmean[f'nu={nu}, len={len_scale}'])
fig, ax = plt.subplots(1,2,figsize=(2*onefigsize[0],onefigsize[1]))   
for var_name in var_names2:
    ax[0].plot(denslist, diffmean[var_name], label=var_name)
    ax[0].fill_between(denslist,diffmin[var_name],diffmax[var_name], alpha = 0.4)
ax[0].set_ylim(bottom=0)
ax[0].set_xlim(0, 0.1)
ax[0].legend(loc='upper left')
ax[0].set_xlabel('Density')
ax[0].set_ylabel('Relative difference')
# plt.savefig(base_path + f'densdiffplot_nodaily_allup_bootstrap_allcorrmethods_w{weighted}_{n_boot}_seed{seed}.pdf')


var_name = 't2m'
if os.path.exists(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt'):
    lenbins, lldistrib, difflldistrib = myload(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')   
else:
    raise NameError(f'LL file not found for {var_name}.')
#fig,ax = plt.subplots()
p = []
for idens in [4,7,-3,-2]: # selects 0.005, 0.01, 0.1
    if idens == 4:
        col = 'tab:blue'
    elif idens == 7:
        col = 'tab:orange'
    elif idens == -3:
        col = 'tab:green'
    else:
        col = 'tab:red'
    dens = denslist[idens]
    n_links = 1#lldistrib[idens,0,:].sum()
    #n_diff = difflldistrib.sum() / len(diffs)
    n_diff = 1#difflldistrib[idens,0,:].sum()
    p1,=ax[1].plot(binstoplot(lenbins),lldistrib[idens,:,:].mean(axis=0)/ n_links, label=f'dens={denslist[idens]}',zorder = -10, color = adjust_lightness(col, amount=1.5))
    p.append(p1)
    ax[1].fill_between(binstoplot(lenbins),[quantile(lldistrib[idens,:,ibin],0.025)/ n_links for ibin in range(lldistrib.shape[2])],[quantile(lldistrib[idens,:,ibin],0.975)/ n_links for ibin in range(lldistrib.shape[2])], color = ax[1].lines[-1].get_color(), alpha = 0.4)
    p2,=ax[1].plot(binstoplot(lenbins),difflldistrib[idens,:,:].mean(axis=0)/ n_diff, label=f'dens={denslist[idens]}', color = adjust_lightness(col, amount=0.5))
    p.append(p2)
    ax[1].fill_between(binstoplot(lenbins),[quantile(difflldistrib[idens,:,ibin],0.025)/ n_diff for ibin in range(difflldistrib.shape[2])],[quantile(difflldistrib[idens,:,ibin],0.975)/ n_diff for ibin in range(difflldistrib.shape[2])], color = ax[1].lines[-1].get_color(), alpha = 0.4)
ax[1].set_xlabel('Link length (in radians)')
ax[1].set_ylabel('Number of links')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
#ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
#ax.set_yscale('log')
#ax.set_xlim(-0.005,0.2)
l = ax[1].legend([(p[0], p[1]),(p[2],p[3]),(p[4], p[5]),(p[6],p[7])], [f'dens={denslist[4]}',f'dens={denslist[7]}',f'dens={denslist[-3]}',f'dens={denslist[-2]}'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
#ax.legend()
ax = enumerate_subplots(ax,fontsize = 16)
plt.savefig(base_path + f'joint_densdiffplot_{var_name}_w{weighted}_{n_boot}_seed{seed}.pdf')


# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
for var_name in var_names:
    if os.path.exists(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt'):
        lenbins, lldistrib, difflldistrib = myload(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')   
    else:
        raise NameError(f'LL file not found for {var_name}.')
    fig,ax = plt.subplots()
    p = []
    for idens in [4,7,-3,-2]: # selects 0.005, 0.01, 0.1
        if idens == 4:
            col = 'tab:blue'
        elif idens == 7:
            col = 'tab:orange'
        elif idens == -3:
            col = 'tab:green'
        else:
            col = 'tab:red'
        dens = denslist[idens]
        n_links = 1#lldistrib[idens,0,:].sum()
        #n_diff = difflldistrib.sum() / len(diffs)
        n_diff = 1#difflldistrib[idens,0,:].sum()
        p1,=ax.plot(binstoplot(lenbins),lldistrib[idens,:,:].mean(axis=0)/ n_links, label=f'dens={denslist[idens]}',zorder = -10, color = adjust_lightness(col, amount=1.5))
        p.append(p1)
        ax.fill_between(binstoplot(lenbins),[quantile(lldistrib[idens,:,ibin],0.025)/ n_links for ibin in range(lldistrib.shape[2])],[quantile(lldistrib[idens,:,ibin],0.975)/ n_links for ibin in range(lldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
        p2,=ax.plot(binstoplot(lenbins),difflldistrib[idens,:,:].mean(axis=0)/ n_diff, label=f'dens={denslist[idens]}', color = adjust_lightness(col, amount=0.5))
        p.append(p2)
        ax.fill_between(binstoplot(lenbins),[quantile(difflldistrib[idens,:,ibin],0.025)/ n_diff for ibin in range(difflldistrib.shape[2])],[quantile(difflldistrib[idens,:,ibin],0.975)/ n_diff for ibin in range(difflldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
    ax.set_xlabel('Link length')
    ax.set_ylabel('Number of links')
    #ax.set_yscale('log')
    #ax.set_xlim(-0.005,0.2)
    l = ax.legend([(p[0], p[1]),(p[2],p[3]),(p[4], p[5]),(p[6],p[7])], [f'dens={denslist[4]}',f'dens={denslist[7]}',f'dens={denslist[-3]}',f'dens={denslist[-2]}'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
    #ax.legend()
    plt.savefig(base_path + f'lldiffplot_asgstats_only2dens_{var_name}_w{weighted}_{n_boot}_seed{seed}.pdf')

# %%
denslist
# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
for var_name in var_names:
    if os.path.exists(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt'):
        lenbins, lldistrib, difflldistrib = myload(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')   
    else:
        raise NameError(f'LL file not found for {var_name}.')
    fig,ax = plt.subplots()
    p = []
    for idens in [2,4,-2]: # selects 0.005, 0.01, 0.1
        if idens == 2:
            col = 'tab:blue'
        elif idens == 4:
            col = 'tab:orange'
        else:
            col = 'tab:green'
        dens = denslist[idens]
        n_links = lldistrib[idens,0,:].sum()
        #n_diff = difflldistrib.sum() / len(diffs)
        n_diff = difflldistrib[idens,0,:].sum()
        lldiffquotient = difflldistrib[idens,:,:] / lldistrib[idens,::2,:]
        p1,=ax.plot(binstoplot(lenbins),lldiffquotient.mean(axis=0), label=f'dens={denslist[idens]}',zorder = -10, color = adjust_lightness(col, amount=1))
        p.append(p1)
        ax.fill_between(binstoplot(lenbins),[quantile(lldiffquotient[:,ibin],0.025) for ibin in range(lldistrib.shape[2])],[quantile(lldiffquotient[:,ibin],0.975) for ibin in range(lldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
        #p2,=ax.plot(binstoplot(lenbins),difflldistrib[idens,:,:].mean(axis=0)/ n_diff, label=f'dens={denslist[idens]}', color = adjust_lightness(col, amount=0.5))
        #p.append(p2)
        #ax.fill_between(binstoplot(lenbins),[quantile(difflldistrib[idens,:,ibin],0.025)/ n_diff for ibin in range(difflldistrib.shape[2])],[quantile(difflldistrib[idens,:,ibin],0.975)/ n_diff for ibin in range(difflldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
    ax.set_xlabel('Link length')
    ax.set_ylabel('Density')
    #ax.set_xlim(-0.005,0.2)
    #l = ax.legend([(p[0], p[1]),(p[2],p[3]),(p[4],p[5])], [f'dens={denslist[2]}',f'dens={denslist[4]}',f'dens={denslist[-2]}'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
    ax.legend()
    plt.savefig(base_path + f'lldiffquotientplot_{var_name}_w{weighted}_{n_boot}_seed{seed}.pdf')
# %%
lldiffquotient.shape

# %%
def get_bootcorr(data, seed,corr_method = 'pearson'):
    np.random.seed(seed)
    boot_idcs = np.random.choice(data.shape[0], data.shape[0], replace = True)
    return compute_empcorr(data[boot_idcs,:],similarity=corr_method)



# if os.path.exists(base_path + f'densdiff_bootstrap_{corr_method}_w{weighted}_{n_boot}_seed{seed}_'+ dataset_nc.split('/',20)[-1] + '.txt'):
#     print(f'Loading densdiff for {var_name}.')
#     dens_diff = myload(base_path + f'densdiff_bootstrap_{corr_method}_w{weighted}_{n_boot}_seed{seed}_'+ dataset_nc.split('/',20)[-1] + '.txt')
#     adjs= []
#     for iboot in range(len(empcorrs)):
#         adjs.append(get_adj(empcorrs[iboot],dens= density,weighted=weighted))
#     diffs = []
#     for i in range(len(adjs)):
#         for j in range(i):
#             diffs.append(np.where(np.abs(adjs[i] - adjs[j])!=0))
#     diffsum = dens_diff[-1]


        # lldistrib = np.zeros((len(lenbins)-1,len(adjs)))
        # # more compute and less saving not using, but dens1 and dens2 adjs
        # for i in range(len(adjs)):
            

        # difflldistrib = np.zeros((len(lenbins)-1,len(diffs)))
        # for i in range(len(diffs)):
        #     difflldistrib[:,i] = np.histogram(dists[diffs[i]],lenbins)[0]
        # mysave(base_path,f'bootstrap_llstats_{corr_method}_w{weighted}_{density}_{n_boot}_seed{seed}_'+dataset_nc.split('/',20)[-1]+'.txt', [lldistrib,difflldistrib])
# %%
denslist