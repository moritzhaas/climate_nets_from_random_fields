# %%
import os
import numpy as np
import matplotlib.pyplot as plt
#from numpy.lib.type_check import nan_to_num
from scipy import signal, stats
# import cartopy as ctp
from climnet.dataset import BaseDataset, AnomalyDataset
from sklearn.linear_model import LinearRegression, Ridge
from climnet.grid import regular_lon_lat, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import *
import time
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})

#plt.rcParams["figure.figsize"] = (8*2,6*2)

# %%
# Grid
#n_time = 100
alpha = 0.05
n_lat = 18 * 4
n_lon = 2 * n_lat
n_time = 1000
num_runs = 10
# grid_stretch = 1
rm_outliers = True
# set parameters
grid_step = 180/n_lat
var_name = 't2m' # 't2m' 'pr' 'sp' 'z'
grid_type = 'fekete'
num_cpus = 48
#lon_range = [-110, 40]
#lat_range = [20, 85]
save = True
# time_range = ['1980-01-01', '2019-12-31']
time_range = None
tm = None  # 'week'
norm = False
detrend = True
absolute = False
denslist = [0.005, 0.01, 0.05, 0.1, 0.2]

PATH = os.path.dirname(os.path.abspath(__file__))
print('You are here: ', PATH)

base_path = '../../climnet_output/' #'~/Documents/climnet_output/'

# %%
# generate grid
grid_stretch = 1
n_lon = 2 * n_lat
grid_step_lon = 360/ n_lon
grid_step_lat = 180/ n_lat
dist_equator = gdistance((0,0),(0,grid_step_lon))
lon2, lat2 = regular_lon_lat(n_lon,n_lat)
regular_grid = {'lon': lon2, 'lat': lat2}
start_date = '2000-01-01'
if os.path.exists(base_path + f'regular_dists_nlat_{n_lat}_nlon_{n_lon}.npy'):
    reg_dists = np.load(base_path +f'regular_dists_nlat_{n_lat}_nlon_{n_lon}.npy')
else:
    reg_dists = all_dists(lat,lon)
    np.save(base_path +f'regular_dists_nlat_{n_lat}_nlon_{n_lon}.npy', reg_dists)

# create fekete grid
num_points = gridstep_to_numpoints(grid_step_lon)
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
if os.path.exists(base_path + f'fekete_dists_npoints_{num_points}.npy'):
    dists = np.load(base_path + f'fekete_dists_npoints_{num_points}.npy')
else:
    dists = np.zeros((len(lon), len(lon)))
    for i in range(len(lon)):
        for j in range(i):
            dists[i,j] = gdistance((lat[i], lon[i]), (lat[j],lon[j]))
            dists[j,i] = dists[i,j]
    np.save(base_path + f'fekete_dists_npoints_{num_points}.npy', dists)

earth_radius = 6371.009
dists /= earth_radius
dist_equator /= earth_radius

seed = int(time.time())
np.random.seed(seed)
# generate igrf data
data = np.zeros((n_time,num_points))
if os.getenv("HOME") == '/Users/moritz':
    #dataset_nc = base_path + f"era5_{var_name}_2.5_ds.nc"
    dataset_nc = base_path + f"era5_{var_name}_1_compr.nc"
else:
    dataset_nc = f"/mnt/qb/goswami/exchange_folder/climate_data/era5_{var_name}_2.5_ds.nc"
# %%
def tail(arr, xs):
    return np.array([(arr>x).sum() for x in xs]) / len(arr)

def gaussian_kernel(x, mean = 0, sig = 1, trunc = 0):
    vals = np.exp(-np.abs(x-mean) ** 2/(2 * sig))
    return np.where(vals <= trunc, 0, vals)

def standardize(dataset, axis=0):
    return (dataset - np.average(dataset, axis=axis)) / (np.std(dataset, axis=axis))


# %%
'''
setname = f'era5_{var_name}_2.5_ds.nc'
da = xr.open_dataset(dataset_nc)
try:
    os.remove(setname)
    da.to_netcdf(setname)
except:
    da.to_netcdf(setname)
'''

ds_name = f'era5_{var_name}_{grid_type}_{grid_step}_tm{tm}_dayofyear_ds.nc'
if os.path.exists(base_path+ds_name):
    ds = AnomalyDataset(load_nc=base_path + ds_name)
else:
    ds = AnomalyDataset(data_nc=dataset_nc, var_name = var_name, grid_type=grid_type, grid_step=grid_step, climatology='dayofyear')
    ds.save(base_path + ds_name)

# %%
ds2 = AnomalyDataset(data_nc=dataset_nc, var_name = var_name, grid_type=grid_type, grid_step=5, climatology='dayofyear')
ds2.save(base_path+f'era5_{var_name}_{grid_type}_{5}_tm{tm}_dayofyear_ds.nc')

if detrend:
    ds2.ds['anomalies'].data = signal.detrend(ds2.ds['anomalies'].data, axis = 0)
    data2 = ds2.ds['anomalies'].data
empcorr2 = np.corrcoef(data2.T)

# %%
ds2monthly = ds2.ds.resample(time='MS').mean('time')
ds2monthly['anomalies'] = ds2.compute_anomalies(ds2monthly[var_name],'month')
# manomalies = ds2.ds['anomalies'].groupby(f"time.month")
# print(manomalies)
data2m = ds2monthly['anomalies'].data
empcorr2m = np.corrcoef(data2m.T)
# %%
if detrend:
    ds.ds['anomalies'].data = signal.detrend(ds.ds['anomalies'].data, axis = 0)
    data = ds.ds['anomalies'].data

if norm:
    # stds = np.tile(ds.ds['anomalies'].data.std(
    #     axis=0), ds.ds['anomalies'].data.shape[0]).reshape(ds.ds['anomalies'].data.shape)
    # ds.ds['anomalies'].data = ds.ds['anomalies'].data / stds
    #ds.ds['anomalies'].data = standardize(ds.ds['anomalies'].data)
    # make input data standard normal:
    data = ds.ds['anomalies'].data
    data_quantiles = (np.argsort(np.argsort(data,axis = 0), axis = 0) + 1)/ (data.shape[0] + 1) #values between 1/(maxrank+2) and (maxrank+1)/(maxrank+2)
    data = stats.norm.ppf(data_quantiles)

print(data.shape)
empcorr = np.corrcoef(data.T)

# %%
#var_name = 't2m'
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
curr_time = time.time()
import cartopy as ctp
idx = 803
#deg_plot = plot_map_lonlat(lon,lat, adj.sum(axis = 0), color=color,label =label, ctp_projection=ctp_projection,vmin=0, grid_step=grid_step_lon, *args)
corr_plot = plot_map_lonlat(ds2.ds.lon.data,ds2.ds.lat.data, empcorr2[idx,:],ctp_projection='EqualEarth', grid_step = 5,vmin =-1,vmax=1, label='Correlation')
corr_plot['ax'].plot(ds2.ds.lon[idx].data, ds2.ds.lat[idx].data, 'o', color = 'green', transform = ctp.crs.PlateCarree())
#plt.title(f'Empirical correlation of temperatures')
#plt.savefig(base_path+f'{var_name}_empcorr{idx}_fekete5_notnorm.png')

corr_plot = plot_map_lonlat(ds2.ds.lon.data,ds2.ds.lat.data, empcorr2m[idx,:],ctp_projection='EqualEarth', grid_step = 5,vmin =-1,vmax=1, label='Correlation')
corr_plot['ax'].plot(ds2.ds.lon[idx].data, ds2.ds.lat[idx].data, 'o', color = 'green', transform = ctp.crs.PlateCarree())
#plt.title(f'Empirical correlation of temperatures')
#plt.savefig(base_path+f'{var_name}_monthly_empcorr{idx}_fekete5_notnorm.png')
# %%
# import cartopy as ctp
# idx =1001
# corr_plot = plot_map_lonlat(ds.ds.lon.data,ds.ds.lat.data, empcorr[idx,:], label='Correlation')
# corr_plot['ax'].plot(ds.ds.lon[idx].data, ds.ds.lat[idx].data, 'o', color = 'green', transform = ctp.crs.PlateCarree())
# plt.title(f'Empirical correlation of temperatures')
# plt.savefig(base_path+f'{var_name}_empcorr{idx}_notnorm.png', dpi = 150)

# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(2)
# create fekete grid
num_points = 1483
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
#lon = ds2.ds.lon.data
#lat = ds2.ds.lat.data
llon = np.concatenate((lon,np.zeros(200)))
llat = np.concatenate((lat,np.linspace(-90,90,200)))
itime = 0
path_realisation = {}
from sklearn.gaussian_process.kernels import Matern
ar_coeff = np.zeros_like(llat)
for nu in [0.5,1.5]:
    for len_scale in [0.1,0.2]:
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        cov = kernel(spherical2cartesian(llon,llat))
        data = diag_var_process(ar_coeff, cov, n_time, n_pre = 200, eps = 1e-8)
        path_realisation[f'{nu}_{len_scale}'] = data[:,-200:]
        if nu == 1.5 and len_scale == 0.2:
            corr_plot = plot_map_lonlat(lon,lat, data[itime,:-200],ctp_projection='EqualEarth', grid_step = 5, label='Anomaly')
            corr_plot['ax'].plot([0,0], [-90, 90], color = 'black', alpha = 1, linestyle = '--', linewidth = 4, transform=ctp.crs.Geodetic())
            #plt.title(f'Empirical correlation of temperatures')
            plt.savefig(base_path+f'rerealisation{itime}_nu{nu}_len{len_scale}.pdf')
        
# %%
nu = 1.5
len_scale=0.2
corr_plot = plot_map_lonlat(lon,lat, data[itime,:-200],ctp_projection='EqualEarth', grid_step = 5, label='Anomaly')
corr_plot['ax'].plot([0,0], [-90, 90], color = 'black', alpha = 1, linestyle = '--', linewidth = 6, transform=ctp.crs.Geodetic())
#plt.title(f'Empirical correlation of temperatures')
plt.savefig(base_path+f'rerealisation{itime}_nu{nu}_len{len_scale}.pdf')
# %%
itime=9#9
fig,ax = plt.subplots()
for nu in [0.5,1.5]:
    for len_scale in [0.1,0.2]:
        if nu == 1.5 and len_scale == 0.2:
            ax.plot(np.linspace(-90,90,200),path_realisation[f'{nu}_{len_scale}'][0,:], label = f'nu={nu}, l={len_scale}')
        else:
            ax.plot(np.linspace(-90,90,200),path_realisation[f'{nu}_{len_scale}'][itime,:], label = f'nu={nu}, l={len_scale}')

ax.set_xlabel('Latitude')
ax.set_ylabel('Anomaly')
ax.legend()
plt.savefig(base_path+f'rerealisation_paths_{itime}.pdf')