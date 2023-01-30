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
num_points, earth_radius = num_points, earth_radius
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
dtele = 5000 / earth_radius # what is a teleconnection? Longer than 5000 km is a cautious choice
#robust_tolerance = ... # 0.5,0.8
threshs = [0.1,0.2,0.3,0.5,0.8]
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
#%%
nbhds2, nbhds3 = {}, {}

nbhdname2 = f'nbhds_{grid_type}{num_points}_eps{eps2}.txt'
if not os.path.exists(base_path+'nbhds/'+nbhdname2):
    for i in range(num_points):
        nbhds2[i] = np.where(dists[i,:] <= eps2)[0]
    mysave(base_path+'nbhds/',nbhdname2, nbhds2)
else:
    nbhds2 = myload(base_path+'nbhds/'+nbhdname2)


        

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
    #denslistplus = deepcopy(denslist)
    #denslistplus.append(density)
    
    # compute decorrelation length, i.e. construct thres net with fixed thresh, for each node find radius at which connect below 1-tau, average
    # compute number of bundles for several dens to find best suited..

    # compute for original
    all_lengths1 = np.zeros((len(threshs),len(dists)))
    all_lengths2 = np.zeros((len(threshs),len(dists)))
    all_countrobusttele2raw1, all_countrobusttele2raw2, all_dens,all_teledens= [np.zeros((len(threshs))) for _ in range(4)]
    for ithres,thresh in enumerate(threshs):
        adj = emp_corr > thresh
        adj[np.eye(adj.shape[0],dtype=bool)] = 0
        density = adj.sum()/((adj.shape[0]-1)*adj.shape[0])
        all_dens[ithres] = density
        # distadj = get_adj(5-dists, density,weighted=True)
        # maxdist = 5-distadj[distadj!=0].min()
        all_lengths1[ithres,:] = decorr_length(adj,dists,min_connectivity=0.8, grid_type=grid_type,base_path=base_path)#bin then save
        all_lengths2[ithres,:] = decorr_length(adj,dists,min_connectivity=0.5, grid_type=grid_type,base_path=base_path)
        # how many telelinks in bundles?
        teleadj = deepcopy(adj)
        teleadj[dists < dtele] = 0
        all_teledens[ithres] = teleadj.sum()/((adj.shape[0]-1)*adj.shape[0])
        if teleadj.sum() == 0:
            all_countrobusttele2raw1[ithres] = 0
            all_countrobusttele2raw2[ithres] = 0
        else:
            all_countrobusttele2raw1[ithres] =  bundlefraction(adj, dists, nbhds2, dtele, tolerance = 0.2, typ = 'raw')
            all_countrobusttele2raw2[ithres] =  bundlefraction(adj, dists, nbhds2, dtele, tolerance = 0.5, typ = 'raw')
    mysave(base_path,f'decorrstats_noboot_{var_name}_dtele{dtele}_{corr_method}_w{weighted}_seed{seed}.txt', [all_dens,all_teledens,all_lengths1,all_lengths2,all_countrobusttele2raw1,all_countrobusttele2raw2])

    # bootstrap data in time
    all_lengths1 = np.zeros((len(threshs),n_boot,len(dists)))
    all_lengths2 = np.zeros((len(threshs),n_boot,len(dists)))
    all_countrobusttele2raw1, all_countrobusttele2raw2, all_dens,all_teledens,all_countdifftele2raw1,all_countdifftele2raw2 = [np.zeros((len(threshs),n_boot)) for _ in range(6)]
    for ithres,thresh in enumerate(threshs):
        for iboot in range(n_boot):
            adj = get_bootcorr(data, seed+2*iboot) > thresh #get_adj(get_bootcorr(data, seed+2*iboot),dens= dens,weighted=weighted)
            adj[np.eye(adj.shape[0],dtype=bool)] = 0
            adj = np.array(adj,dtype=np.float64)
            density = adj.sum()/((adj.shape[0]-1)*adj.shape[0])
            all_dens[ithres,iboot] = density
            # distadj = get_adj(5-dists, density,weighted=True)
            # maxdist = 5-distadj[distadj!=0].min()
            all_lengths1[ithres,iboot,:] = decorr_length(adj,dists,min_connectivity=0.8, grid_type=grid_type,base_path=base_path)#bin then save
            all_lengths2[ithres,iboot,:] = decorr_length(adj,dists,min_connectivity=0.5, grid_type=grid_type,base_path=base_path)
            # how many telelinks in bundles?
            teleadj = deepcopy(adj)
            teleadj[dists < dtele] = 0
            all_teledens[ithres,iboot] = teleadj.sum()/((adj.shape[0]-1)*adj.shape[0])
            if teleadj.sum() == 0:
                all_countrobusttele2raw1[ithres,iboot] = 0
                all_countrobusttele2raw2[ithres,iboot] = 0
            else:
                all_countrobusttele2raw1[ithres,iboot] =  bundlefraction(adj, dists, nbhds2, dtele, tolerance = 0.2, typ = 'raw')
                all_countrobusttele2raw2[ithres,iboot] =  bundlefraction(adj, dists, nbhds2, dtele, tolerance = 0.5, typ = 'raw')
            adj2 = get_bootcorr(data, seed+2*iboot+1) > thresh
            adj2[np.eye(adj.shape[0],dtype=bool)] = 0
            diff = np.where(np.logical_and(np.abs(adj - adj2)!=0,dists>dtele))
            # compute how many differing links are part of some bundle
            all_countdifftele2raw1[ithres,iboot] = bundlefractionwhere(adj, diff, nbhds2, tolerance = 0.2, typ = 'raw')
            all_countdifftele2raw2[ithres,iboot] = bundlefractionwhere(adj, diff, nbhds2, tolerance = 0.5, typ = 'raw')
            

    mysave(base_path,f'decorrstats_{var_name}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt', [all_dens,all_teledens,all_lengths1,all_lengths2,all_countrobusttele2raw1,all_countrobusttele2raw2,all_countdifftele2raw1,all_countdifftele2raw2])

# %%
decorrstats = {}
decorrstatsnoboot = {}
for var_name in var_names:
    decorrstats[var_name] = myload(base_path+f'decorrstats_{var_name}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')
    decorrstatsnoboot[var_name] = myload(base_path+f'decorrstats_noboot_{var_name}_dtele{dtele}_{corr_method}_w{weighted}_seed{seed}.txt')

# all_dens,all_teledens,all_lengths1,all_lengths2,all_countrobusttele2raw1,all_countrobusttele2raw2

# %%
# TODO: calc all_countrobusttele2raw1 and all_countdifftele2raw1 for MIGRF

'''
var_names2 = deepcopy(var_names)
nus = [0.5,1.5]
len_scales = [0.1,0.2]
for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        var_names2.append(f'nu={nu}, len={len_scale}')

for inu,nu in enumerate(nus):
    for ilen,len_scale in enumerate(len_scales):
        var_name = f'nu={nu}, len={len_scale}'
        decorrstats[var_name] = myload(base_path+?TODO
'''

# %%
def text(x, y, text):
    ax.text(x, y, text, backgroundcolor="white",
            ha='center', va='top', weight='bold', color='blue')

# first plot avg decorr length against links in bundles
for ithres in range(len(threshs)): # one plot for each thresh
    fig,ax = plt.subplots()
    for var_name in var_names: # avg across iboot and node
        bundlefrac = decorrstatsnoboot[var_name][4]# / (decorrstatsnoboot[var_name][1] * ((num_points-1)*num_points/2))
        ax.scatter(decorrstatsnoboot[var_name][2][ithres,:].mean(), bundlefrac[ithres],label=var_name,marker='+')
        #text(decorrstatsnoboot[var_name][2][ithres,:].mean(), bundlefrac[ithres], var_name)
    ax.legend()
    ax.set_xlabel('Decorrelation length')
    ax.set_ylabel('Links in bundle')
    plt.savefig(base_path + f'decorr_bundle_noboot_cross_minconn0.8_tol0.2_thres{threshs[ithres]}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.pdf')

# with more tolerance
for ithres in range(len(threshs)): # one plot for each thresh
    fig,ax = plt.subplots()
    for var_name in var_names2: # avg across iboot and node
        bundlefrac = decorrstatsnoboot[var_name][5]# / (decorrstatsnoboot[var_name][1] * ((num_points-1)*num_points/2))
        ax.scatter(decorrstatsnoboot[var_name][2][ithres,:].mean(), bundlefrac[ithres],label=var_name)
    ax.legend()
    ax.set_xlabel('Decorrelation length')
    ax.set_ylabel('Links in bundle')
    plt.savefig(base_path + f'decorr_bundle_noboot_cross_minconn0.8_tol0.5_thres{threshs[ithres]}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.pdf')



# %%
# first plot avg decorr length against links in bundles
for ithres in range(len(threshs)): # one plot for each thresh
    fig,ax = plt.subplots()
    for var_name in var_names: # avg across iboot and node
        bundlefrac = decorrstats[var_name][-2]#this has been accounted for!! / (decorrstats[var_name][1] * ((num_points-1)*num_points/2))
        ax.scatter(decorrstats[var_name][2][ithres,:,:].mean(), bundlefrac[ithres,:].mean(),label=var_name)
    ax.legend()
    ax.set_xlabel('Decorrelation length')
    ax.set_ylabel('Diff. links in bundle')
    plt.savefig(base_path + f'decorr_bundle_minconn0.8_tol0.2_thres{threshs[ithres]}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.pdf')

# with more tolerance
for ithres in range(len(threshs)): # one plot for each thresh
    fig,ax = plt.subplots()
    for var_name in var_names: # avg across iboot and node
        bundlefrac = decorrstats[var_name][-1]# / (decorrstats[var_name][1] * ((num_points-1)*num_points/2))
        ax.scatter(decorrstats[var_name][2][ithres,:,:].mean(), bundlefrac[ithres,:].mean(),label=var_name)
    ax.legend()
    ax.set_xlabel('Decorrelation length')
    ax.set_ylabel('Links in bundle')
    plt.savefig(base_path + f'decorr_bundle_minconn0.8_tol0.5_thres{threshs[ithres]}_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.pdf')


# %%
# plot teledens: how many telelinks for different vars and thresh?
# xaxis: thresh, yaxis: dark: teledens, light: dens
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import matplotlib.colors as mcolors
colors = []
for col in mcolors.TABLEAU_COLORS:
    colors.append(col)

fig,ax = plt.subplots()
p = []
for ivar,var_name in enumerate(var_names):
    p1,=ax.plot(threshs,decorrstatsnoboot[var_name][0], label=var_name, color = adjust_lightness(colors[ivar], amount=0.5))
    p.append(p1)
    #ax.fill_between(binstoplot(lenbins),[quantile(lldistrib[idens,:,ibin],0.025)/ n_links for ibin in range(lldistrib.shape[2])],[quantile(lldistrib[idens,:,ibin],0.975)/ n_links for ibin in range(lldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
    p2,=ax.plot(threshs,decorrstatsnoboot[var_name][1], label=var_name, color = adjust_lightness(colors[ivar], amount=1.5))
    p.append(p2)
    #ax.fill_between(binstoplot(lenbins),[quantile(difflldistrib[idens,:,ibin],0.025)/ n_links for ibin in range(difflldistrib.shape[2])],[quantile(difflldistrib[idens,:,ibin],0.975)/ n_links for ibin in range(difflldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
ax.set_xlabel('Threshold')
ax.set_ylabel('Network density')
#ax.set_xlim(-0.005,0.2)
ps = []
for ivar in range(len(var_names)):
    ps.append((p[int(2*ivar)],p[int(2*ivar+1)]))
l = ax.legend(ps, var_names, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
#ax.legend()
plt.savefig(base_path + f'teledensplot_{corr_method}_w{weighted}.pdf')
# %%
# plot lengths map and histogram
decorrbins = np.linspace(0,np.pi/2)
for var_name in var_names:
    for ithres, thresh in enumerate(threshs):
        plt.hist(decorrstatsnoboot[var_name][2][ithres,:],decorrbins)
        plt.xlabel('Decorrelation length')
        plt.savefig(base_path+f'decorrlen_hist_tol0.2_{var_name}_th{thresh}.pdf')
        plt.clf()
        plt.hist(decorrstatsnoboot[var_name][3][ithres,:],decorrbins)
        plt.xlabel('Decorrelation length')
        plt.savefig(base_path+f'decorrlen_hist_tol0.5_{var_name}_th{thresh}.pdf')
        plt.clf()
        plot_map_lonlat(grid.grid['lon'],grid.grid['lat'],decorrstatsnoboot[var_name][2][ithres,:],vmin=0,vmax=np.pi/4,color='Reds',ctp_projection='EqualEarth',label='Decorrel. length',earth = True)
        plt.savefig(base_path+f'decorrlen_map_tol0.2_{var_name}_th{thresh}.pdf')
        plt.clf()
        plot_map_lonlat(grid.grid['lon'],grid.grid['lat'],decorrstatsnoboot[var_name][3][ithres,:],vmin=0,vmax=np.pi/4,color='Reds',ctp_projection='EqualEarth',label='Decorrel. length',earth = True)
        plt.savefig(base_path+f'decorrlen_map_tol0.5_{var_name}_th{thresh}.pdf')
        plt.clf()

# %%
max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')
adj = get_adj(emp_corr, density, weighted=False)
print(adj.sum(axis=1).max())
# %%
longlinks = np.where(np.logical_and(adj != 0, dists > max_dist))
1/len(longlinks[0])
# %%
