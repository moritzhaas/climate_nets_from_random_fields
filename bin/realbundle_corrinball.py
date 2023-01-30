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
shortdenslist = [0.001,0.01,0.05,0.1,0.2]
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
seed = 1204
timeparts = 10
density = 0.01

dtele = 5000 / earth_radius # what is a teleconnection? Longer than 5000 km is a cautious choice
#robust_tolerance = ... # 0.5,0.8
threshs = [0.1,0.2,0.3,0.5,0.8]  #lendens = [0.005,0.01,0.05,0.1]

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
    #all_lengths1 = np.zeros((len(shortdenslist),len(dists)))
    #all_lengths2 = np.zeros((len(shortdenslist),len(dists)))
    #all_countrobusttele2raw1, all_countrobusttele2raw2, all_thresh,all_teledens= [np.zeros((len(shortdenslist))) for _ in range(4)]
    
    avgcorr2 = np.zeros(num_points)
    for pt in range(num_points):
        avgcorr2[pt] = emp_corr[pt,nbhds2[pt]][emp_corr[pt,nbhds2[pt]]!=1].mean()
    mysave(base_path,f'avglocalcorr2_{var_name}_{corr_method}.txt', avgcorr2)

    # bootstrap data in time
    avgcorr2 = np.zeros((num_points,n_boot))
    for iboot in range(n_boot):
        bootcorr = get_bootcorr(data, seed+2*iboot) #get_adj(get_bootcorr(data, seed+2*iboot),dens= dens,weighted=weighted)
        for pt in range(num_points):
            avgcorr2[pt,iboot] = bootcorr[pt,nbhds2[pt]][bootcorr[pt,nbhds2[pt]]!=1].mean()
    mysave(base_path,f'avglocalcorr2_boot_{var_name}_{corr_method}.txt', avgcorr2)

# %%
for var_name in var_names:
    avgcorr2=myload(base_path+f'avglocalcorr2_boot_{var_name}_{corr_method}.txt')
    print(var_name, avgcorr2.mean())
# %%
