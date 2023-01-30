# %%
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
from climnet.myutils import *
import time
from climnet.grid import FeketeGrid, regular_lon_lat
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()

#plt.rcParams["figure.figsize"] = (8*2,6*2)
#plt.style.use('bmh')



num_runs = 30
# Network parameters
n_lat = 18 * 4 # makes resolution of 180 / n_lat degrees
n_time = 100
ar = 0
ar2 = None
distrib = 'igrf'
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = False
ranks = False
corr_method='spearman' # 'spearman', 'MI', 'HSIC', 'ES'
robust_tolerance = 0.5
denslist = [0.001,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
ks = [6, 60, 300, 600,1200] #[5,  10, 65, 125, 250]

if len(denslist) != len(ks) and typ == 'threshold':
    raise RuntimeError('Denslist needs to have same length as ks.')

# covariance parameters
var = 10
nus = [0.5, 1.5, 2.5]
len_scales = [0.01,0.05,0.1,0.2] #range to proportions of radius of sphere:  0.01, 0.05, 0.1, 0.2

#nu1 = 0.419 # ozone data
#nu2 = 1
#nu3 = 1.46 #climate model data
#A = np.array([legendre_matern(i,nu,alphasq=alphasq) for i in range(K+1)])

base_path = '../../climnet_output/'

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
#dists /= earth_radius
#dist_equator /= earth_radius
# if not weighted:
#     ranks = False

# diagonal VAR1 coeff of length num_points
ar_coeff = ar * np.ones(num_points)
if ar2 is not None:
    rd_idcs = np.random.permutation(np.arange(num_points))[:num_points // 2]
    ar_coeff[rd_idcs] = ar2 * np.ones(len(rd_idcs))

# name = f'matern{nu}_{len_scale}_{distrib}_{corr_method}_w{weighted}_r{ranks}'
# # def thisname and replace filename
# if ar_coeff is None:
#     thisname = name + f'_{typ}_{grid_type}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}'
# else:
#     if ar2 is None:
#         thisname = name + f'_{typ}_{grid_type}_ar{ar}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}'
#     else:
#         thisname = name + f'_{typ}_{grid_type}_ar{ar}vs{ar2}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}'
# filename = base_path + thisname +'.nc'

print('Producing data.')
curr_time = time.time()
cartesian_grid = spherical2cartesian(lon,lat)
# %%
for nu in nus:
    for len_scale in len_scales:
        # generate data with chordal Matern covariance
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        cov = kernel(cartesian_grid)
        for irun in range(num_runs):
            seed = int(time.time())
            np.random.seed(seed)
            data = diag_var_process(ar_coeff, cov, n_time)
            mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',data)
            #np.save(base_path +f'empdata/data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_var{var}_seed{seed}.npy', data)


# %%
kernel = 1.0 * Matern(length_scale=0.1, nu=0.5)
cov = kernel(cartesian_grid)

ks = [6, 60, 300, 600,1200]
print('Under nu=1.5, len_scale = 0.1 (does not play a role): ')
for k in ks:
    adj = knn_adj(cov, k, weighted=False)
    print(f'Unweighted {k}NN graph has density ', adj.sum() / ((len(adj)-1)*len(adj)))