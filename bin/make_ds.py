# %%
import os
import numpy as np
import matplotlib.pyplot as plt
#from numpy.lib.type_check import nan_to_num
from scipy import signal, stats
# import cartopy as ctp
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

def tail(arr, xs):
    return np.array([(arr>x).sum() for x in xs]) / len(arr)

def gaussian_kernel(x, mean = 0, sig = 1, trunc = 0):
    vals = np.exp(-np.abs(x-mean) ** 2/(2 * sig))
    return np.where(vals <= trunc, 0, vals)

def standardize(dataset, axis=0):
    return (dataset - np.average(dataset, axis=axis)) / (np.std(dataset, axis=axis))

print('HOME is: ', os.getenv("HOME"))
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='spearman'
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True

var_name = 't2m' # ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr'] #, 'sp'
var_names = ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr', 'sp']
ar = 0
ar2 = None
var = 10

num_runs = 2#30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 0.5
len_scale = 0.1

denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]
#robust_tolerance = 0.5

filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'


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



# %%
# Grid
grid_step = 2.5
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


seed = int(time.time())
np.random.seed(seed)
# generate igrf data
data = np.zeros((n_time,num_points))

base_path = '../../climnet_output/'
base_path = base_path + 'real/'
if not os.path.exists(base_path):
    os.mkdir(base_path)


import xarray as xr
from climnet.dataset_new import AnomalyDataset


for var_name in var_names:
    # if (os.getenv("HOME") == '/Users/moritz') or (os.getenv("HOME") is None):    
    #     dataset_nc = base_path + f"era5_singlelevel_monthly_temp2m_1979-2020.nc"
    #     #dataset_nc = '/Volumes/backups2/2m_temperature_sfc_1979_2020.nc'
    # else:# ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr'] #, 'sp'
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

    if not os.path.exists(mydataset_nc):
        print('Create Dataset')
        ds = AnomalyDataset(data_nc=dataset_nc, var_name=None, grid_step=grid_step,
                            grid_type=grid_type, detrend=True,
                            climatology="month")
        ds.save(mydataset_nc)
        print(f"Saved dataset for {list(ds.ds.data_vars.keys())}")
    else:
        ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False)

    data = ds.ds['anomalies'].data
    