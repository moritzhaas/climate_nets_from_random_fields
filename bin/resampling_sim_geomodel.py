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
from climnet.dataset_new import AnomalyDataset
start_time = time.time()
curr_time = time.time()
# from tueplots import bundles
# plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update({"figure.dpi": 300})


irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', irun)
start_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

var_name = 't2m'
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
if weighted:
    robust_tolerance = 0.5
else:
    robust_tolerance = 0.2

#num_surrogates = 10#0

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 1.5 # irrelevant
len_scale = 0.2 # irrelevant

denslist = [0.001,0.005,0.01,0.05,0.1]
ks = [6, 60, 300, 600,1200]


filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

# %%
#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

epsgeo = 0.05 # choose smartly?
n_rewire = 10 # * number of links
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

all_lls = np.zeros((len(denslist), all_lls.shape[1]))

num_links,all_counttele2,all_countrobusttele2,all_countrobusttele2raw= [np.zeros((len(denslist))) for _ in range(4)]

all_geo_lls, all_boot_lls = [np.zeros((len(denslist),all_lls.shape[1], num_runs)) for _ in range(2)]

all_geo_llquant1,all_geo_llquant2,all_geo_counttele2,all_geo_countrobusttele2,all_geo_countrobusttele2raw,all_boot_llquant1,all_boot_llquant2,all_boot_counttele2,all_boot_countrobusttele2,all_boot_countrobusttele2raw = [np.zeros((len(denslist),num_runs)) for _ in range(5*2)]

grid_step = 180 / n_lat


if (os.getenv("HOME") == '/Users/moritz') or (os.getenv("HOME") is None):    
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


mydataset_nc = '../../climnet_output/real/data_fekete2.5_month_detrended_era5_singlelevel_monthly_temp2m_1979-2020.nc'


ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = grid_step,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)
    
# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'

import sys
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# %%
for i, density in enumerate(denslist):
    #print('Computing tele. run, idens, time since previous print: ',irun,i, time.time()-curr_time)
    print(f'Starting density {density}. Time: {time.time()-curr_time}')
    curr_time = time.time()
    if typ == 'threshold':
        adj = get_adj(emp_corr, density,weighted=weighted)
    else:
        adj = knn_adj(np.abs(emp_corr),ks[i],weighted = weighted)
    if weighted and ranks:
        adj = rank_matrix(adj)
        adj /= (adj.max()+1)
    # deg = adj.sum(axis = 0)/(adj.shape[0]-1)
    num_links[i] = (adj!=0).sum() / 2
    # lls = dists * adj
    # sorted_lls = np.sort((lls)[lls != 0])
    # all_lls[i,: ] = np.histogram(sorted_lls, bins = llbins)[0]
    # if len(sorted_lls) == 0:
    #     all_llquant1[i ] = 0
    #     all_llquant2[i ] = 0
    # else:
    #     all_llquant1[i ] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
    #     all_llquant2[i ] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]
    
    # distadj = get_adj(5-dists, density,weighted=True)
    # maxdist = 5-distadj[distadj!=0].min()
    # all_counttele2[i] = bundlefraction(adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = '1tm')
    # all_countrobusttele2[i] = bundlefraction(adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'lw')
    # all_countrobusttele2raw[i] =  bundlefraction(adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'raw')

    # curr_time = time.time()
    # G = nx.from_numpy_matrix(adj)
    # cc = nx.clustering(G) #, nodes = idcs)
    # ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
    # all_ccs[i,: ] = ccnum/num_points

    #geo_adj = GeoNetwork(geogrid, adjacency=adj, directed=False, node_weight_type='surface', silence_level=0)
    #for inet in range(num_surrogates):   
    inet = irun     
    # compute Geomodel II surrogate
    geo_adj = geomodel2(adj,dists,epsgeo,n_iter=np.minimum(1e7,num_links[i] * n_rewire))
    mysave(base_path+'geo/',f'geomodeladj_dens{density}_geps{epsgeo}_rew{n_rewire}_nlat{n_lat}_run{irun}.txt',geo_adj)
    
    #geo_adj.randomly_rewire_geomodel_II(dists,iterations=num_links[i], inaccuracy=0.1) # TODO: something betw 0.01 and 0.25, may be important

    # compute bootstrap surrogate
    # bootsample = np.random.choice(n_time,n_time)
    # boot_adj = get_adj(compute_empcorr(data[bootsample,:],similarity=corr_method),density,weighted=weighted)

    # # compute link length distirbution for each surrogate (mean and std on each bin)
    # geo_lls = dists * geo_adj
    # sorted_lls = np.sort((geo_lls)[geo_lls != 0])
    # all_geo_lls[i, :,inet] = np.histogram(sorted_lls, bins = llbins)[0]
    # if len(sorted_lls) == 0:
    #     all_geo_llquant1[i, inet] = 0
    #     all_geo_llquant2[i, inet] = 0
    # else:
    #     all_geo_llquant1[i, inet] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
    #     all_geo_llquant2[i, inet] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

    # geo_lls = dists * geo_adj TODO
    # sorted_lls = np.sort((geo_lls)[geo_lls != 0])
    # all_geo_lls[i, inet, :] = np.histogram(sorted_lls, bins = llbins)[0]
    # if len(sorted_lls) == 0:
    #     all_geo_llquant1[i, inet] = 0
    #     all_geo_llquant2[i, inet] = 0
    # else:
    #     all_geo_llquant1[i, inet] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
    #     all_geo_llquant2[i, inet] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

    # boot_lls = dists * boot_adj
    # sorted_lls = np.sort((boot_lls)[boot_lls != 0])
    # all_boot_lls[i, :,inet] = np.histogram(sorted_lls, bins = llbins)[0]
    # if len(sorted_lls) == 0:
    #     all_boot_llquant1[i, inet] = 0
    #     all_boot_llquant2[i, inet] = 0
    # else:
    #     all_boot_llquant1[i, inet] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
    #     all_boot_llquant2[i, inet] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

    # count tele
    # distadj = get_adj(5-dists, density,weighted=True)
    # maxdist = 5-distadj[distadj!=0].min()
    # all_geo_counttele2[i, inet] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = '1tm')
    # all_geo_countrobusttele2[i,inet] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'lw')
    # all_geo_countrobusttele2raw[i,inet] =  bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'raw')

    # all_boot_counttele2[i, inet] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = '1tm')
    # all_boot_countrobusttele2[i,inet] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'lw')
    # all_boot_countrobusttele2raw[i,inet] =  bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'raw')
    
    # same for cc
    # curr_time = time.time()
    # G = nx.from_numpy_matrix(iaaft_adj)
    # cc = nx.clustering(G) #, nodes = idcs)
    # ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
    # all_iaaft_ccs[i,inet,:] = ccnum/num_points

    # G = nx.from_numpy_matrix(geo_adj)
    # cc = nx.clustering(G) #, nodes = idcs)
    # ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
    # all_geo_ccs[i,inet,:] = ccnum/num_points

    # G = nx.from_numpy_matrix(boot_adj)
    # cc = nx.clustering(G) #, nodes = idcs)
    # ccnum = np.histogram(list(cc.values()),bins = degbins)[0]
    # all_boot_ccs[i,inet,:] = ccnum/num_points
    
    # degree for fixed node, when not fixed by resampling procedure..


    # outfilename = f'all_resampling_geo_run{inet}_dens{density}_eps{epsgeo}_{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.txt'
    # all_stats = [num_links,all_lls,all_llquant1, all_llquant2, all_counttele2, all_countrobusttele2, all_countrobusttele2raw,all_geo_lls,all_geo_llquant1, all_geo_llquant2, all_geo_counttele2, all_geo_countrobusttele2, all_geo_countrobusttele2raw]#,all_boot_lls,all_boot_llquant1, all_boot_llquant2, all_boot_counttele2, all_boot_countrobusttele2, all_boot_countrobusttele2raw] # all_tele3, all_robusttele3,
    # mysave(base_path,outfilename,all_stats)
# all_stats = [all_dens,all_tele1,all_tele2, all_robusttele2, all_mad, all_shufflemad] # all_tele3, all_robusttele3,
# with open(base_path + outfilename, "wb") as fp:   #Pickling
#     pickle.dump(all_stats, fp)

# %%
# from pyunicorn.core.geo_network import GeoNetwork
# from pyunicorn.core.grid import Grid
# time_seq = np.zeros_like(lat)
# geogrid= Grid(time_seq, lat, lon, silence_level=0)
# net = GeoNetwork.SmallTestNetwork()
# net.randomly_rewire_geomodel_I(
#         distance_matrix=net.grid.angular_distance(),
#         iterations=100, inaccuracy=1.0)
# %%
# import pyunicorn
#GeoNetwork.SmallTestNetwork()

# # %%
# geps = 0.05
# rew = 10
# for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
#     geoadjlist = find(f'geomodeladj_dens{density}*geps{geps}_rew{rew}_*',base_path)
#     all_geo_lls = np.zeros((5,all_lls.shape[1],len(geoadjlist)))
#     all_geo_llquant1,all_geo_llquant2,all_geo_counttele2,all_geo_countrobusttele2,all_geo_countrobusttele2raw = [np.zeros((5,len(geoadjlist))) for _ in range(5)]
#     for filename in geoadjlist:
#         run = int(filename.split('_run',1)[1].split('.txt')[0])
#         geo_adj = myload(filename)
#         geo_lls = dists * geo_adj
#         sorted_lls = np.sort((geo_lls)[geo_lls != 0])
#         all_geo_lls[i, :,run] = np.histogram(sorted_lls, bins = llbins)[0]
#         if len(sorted_lls) == 0:
#             all_geo_llquant1[i, run] = 0
#             all_geo_llquant2[i, run] = 0
#         else:
#             all_geo_llquant1[i, run] = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
#             all_geo_llquant2[i, run] = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

#         # count tele
#         distadj = get_adj(5-dists, density,weighted=True)
#         maxdist = 5-distadj[distadj!=0].min()
#         all_geo_counttele2[i, run] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = '1tm')
#         all_geo_countrobusttele2[i,run] = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'lw')
#         all_geo_countrobusttele2raw[i,run] =  bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'raw')

#         #np.float64(graphstatname.split('nu',1)[1].split('_',1)[0])
#         #geomodeladj_dens0.01_geps0.05_rew10_nlat72_run3
#     mysave(base_path,f'geostats_runs{len(geoadjlist)}_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}.txt')



# # %%
# fig,ax = plt.subplots()
# for lineidx in range(num_lines):
#     ax.plot(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
#     ax.fill_between(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1) - 2 *all_lls[lineidx,:,:].std(axis = 1), all_lls[lineidx,:,:].mean(axis = 1) + 2 *all_lls[lineidx,:,:].std(axis = 1), alpha = 0.4)
#     ax.plot(all_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
#     ax.plot(all_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
#     upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
#     ax.plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax.lines[-1].get_color())
# ax.legend()
# ax.set_ylabel(f'Number of links')
# ax.set_xlabel(f'Distance (in radians)')
# ymax = all_lls.max()
# ax.set_ylim(-0.1*ymax,1.1*ymax)
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
# #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
# ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
# plt.savefig(base_path+'graphstatplots/plot_llresampl_'+savename+'.pdf')



# # %%
# # %%
# from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

# var_name = 't2m'

# if os.path.exists(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt'):
#     lenbins, lldistrib, difflldistrib = myload(base_path + f'difflldistrib_bootstrap_{var_name}_{corr_method}_w{weighted}_{n_boot}_seed{seed}.txt')   
# else:
#     raise NameError(f'LL file not found for {var_name}.')
# fig,ax = plt.subplots()
# p = []
# idens = 4 #[2,4,-2]: # selects 0.005, 0.01, 0.1
# if idens == 2:
#     col = 'tab:blue'
# elif idens == 4:
#     col = 'tab:orange'
# else:
#     col = 'tab:green'
# dens = denslist[idens]
# n_links = lldistrib[idens,0,:].sum()
# #n_diff = difflldistrib.sum() / len(diffs)
# n_diff = difflldistrib[idens,0,:].sum()
# lldiffquotient = difflldistrib[idens,:,:] / lldistrib[idens,::2,:]
# p1,=ax.plot(binstoplot(lenbins),lldistrib.mean(axis=0), label=f'dens={denslist[idens]}',zorder = -10, color = adjust_lightness(col, amount=1))
# p.append(p1)
# ax.fill_between(binstoplot(lenbins),[quantile(lldiffquotient[:,ibin],0.025) for ibin in range(lldistrib.shape[2])],[quantile(lldiffquotient[:,ibin],0.975) for ibin in range(lldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
# #p2,=ax.plot(binstoplot(lenbins),difflldistrib[idens,:,:].mean(axis=0)/ n_diff, label=f'dens={denslist[idens]}', color = adjust_lightness(col, amount=0.5))
# #p.append(p2)
# #ax.fill_between(binstoplot(lenbins),[quantile(difflldistrib[idens,:,ibin],0.025)/ n_diff for ibin in range(difflldistrib.shape[2])],[quantile(difflldistrib[idens,:,ibin],0.975)/ n_diff for ibin in range(difflldistrib.shape[2])], color = ax.lines[-1].get_color(), alpha = 0.4)
# ax.set_xlabel('Link length')
# ax.set_ylabel('Density')
# #ax.set_xlim(-0.005,0.2)
# #l = ax.legend([(p[0], p[1]),(p[2],p[3]),(p[4],p[5])], [f'dens={denslist[2]}',f'dens={denslist[4]}',f'dens={denslist[-2]}'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
# ax.legend()
# plt.savefig(base_path + f'lldiffquotientplot_{var_name}_w{weighted}_{n_boot}_seed{seed}.pdf')