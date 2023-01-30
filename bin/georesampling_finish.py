# %%
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
grid_step = 180 / n_lat



# %%
geps = 0.05
rew = 10
for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
    geoadjlist = find(f'geomodeladj_dens{density}*geps{geps}_rew{rew}_*',base_path+'geo/')
    all_geo_lls = np.zeros((5,all_lls.shape[1],len(geoadjlist)))
    all_geo_llquant1,all_geo_llquant2,all_geo_counttele2,all_geo_countrobusttele2,all_geo_countrobusttele2raw = [np.zeros((5,len(geoadjlist))) for _ in range(5)]
    for filename in geoadjlist:
        thisrun = int(filename.split('_run',1)[1].split('.txt')[0])
        if thisrun != irun:
            continue
        geo_adj = myload(filename)
        geo_lls = dists * geo_adj
        sorted_lls = np.sort((geo_lls)[geo_lls != 0])
        all_geo_lls = np.histogram(sorted_lls, bins = llbins)[0]
        if len(sorted_lls) == 0:
            all_geo_llquant1 = 0
            all_geo_llquant2 = 0
        else:
            all_geo_llquant1 = sorted_lls[int(np.ceil(alpha1 * len(sorted_lls)-1))]
            all_geo_llquant2 = sorted_lls[int(np.ceil(alpha2 * len(sorted_lls)-1))]

        # count tele
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_geo_counttele2 = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = '1tm')
        all_geo_countrobusttele2 = bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'lw')
        all_geo_countrobusttele2raw =  bundlefraction(geo_adj, dists, nbhds2, maxdist, tolerance = robust_tolerance, typ = 'raw')

        #np.float64(graphstatname.split('nu',1)[1].split('_',1)[0])
        #geomodeladj_dens0.01_geps0.05_rew10_nlat72_run3
    all_stats = [all_geo_lls,all_geo_llquant1, all_geo_llquant2, all_geo_counttele2, all_geo_countrobusttele2, all_geo_countrobusttele2raw]
    mysave(base_path+'geo/',f'geostats_run{irun}_runs{len(geoadjlist)}_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}.txt',all_stats)

# %%
# all_lls = np.zeros((len(denslist), all_lls.shape[1]))

# num_links,all_counttele2,all_countrobusttele2,all_countrobusttele2raw= [np.zeros((len(denslist))) for _ in range(4)]

# all_geo_lls, all_boot_lls = [np.zeros((len(denslist),all_lls.shape[1], num_runs)) for _ in range(2)]

# all_geo_llquant1,all_geo_llquant2,all_geo_counttele2,all_geo_countrobusttele2,all_geo_countrobusttele2raw,all_boot_llquant1,all_boot_llquant2,all_boot_counttele2,all_boot_countrobusttele2,all_boot_countrobusttele2raw = [np.zeros((len(denslist),num_runs)) for _ in range(5*2)]

# for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
#     for irun in range(30):
#         filename = find(f'geostats_run{irun}_*_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}*',base_path)[0]
#         one_geo_lls,one_geo_llquant1, one_geo_llquant2, one_geo_counttele2, one_geo_countrobusttele2, one_geo_countrobusttele2raw = myload(filename)
#         all_geo_lls[i, :,irun] = one_geo_lls