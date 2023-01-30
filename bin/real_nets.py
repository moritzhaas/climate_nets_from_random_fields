# %%
from ast import arg
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap#from numpy.lib.type_check import nan_to_num
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
dtele = 5000 / earth_radius # what is a teleconnection? Longer than 5000 km is a cautious choice
#robust_tolerance = ... # 0.5,0.8
threshs = [0.1,0.2,0.3,0.5,0.8]
seed = 1204
timeparts = 10
density = 0.005
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

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

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

def plot_deg_with_edges(grid, adj, transform = ctp.crs.PlateCarree(),label = None,  ctp_projection='EqualEarth',vmin = None,vmax =None,  color ='Reds', pts = None, *args):
    if vmax is None:
        deg_plot = plot_map_lonlat(lon,lat, adj.sum(axis = 0), color=color,label =label, ctp_projection=ctp_projection,vmin=0, grid_step=grid_step_lon, *args)
    else:
        deg_plot = plot_map_lonlat(lon,lat, adj.sum(axis = 0), color=color,label =label, ctp_projection=ctp_projection,vmin=vmin,vmax=vmax, grid_step=grid_step_lon, *args)
    if pts is None:
        edges = np.where(adj != 0)
        for i in range(len(edges[0])):
            deg_plot['ax'].plot([grid['lon'][edges[0][i]],grid['lon'][edges[1][i]]], [grid['lat'][edges[0][i]], grid['lat'][edges[1][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
    else:
        for point in pts:
            edges = np.where(adj[point,:] != 0)
            for i in range(len(edges[0])):
                deg_plot['ax'].plot([grid['lon'][point], grid['lon'][edges[0][i]]], [grid['lat'][point], grid['lat'][edges[0][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
    return deg_plot

# %%
import matplotlib
import matplotlib.colors as cl
adjust_fontsize(3)
bounds = np.linspace(-1,1,100)
norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
norms = [norm,None,None,norm,None,None]

data_list=[]
vmins,vmaxs=[],[]
extends = ['neither','max',None,'neither','max',None]
labels = ['Autocorrelation','Degree',None,'Autocorrelation','Degree',None]
colors=['RdBu_r','Reds',None,'RdBu_r','Reds',None]
tickss=[np.linspace(-1,1,9),None,None,np.linspace(-1,1,9),None,None]

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
var_name = 't2m'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
degs=adj.sum(axis=1)

ar_grid = np.diag(autocorr(data))

vmins.append(ar_grid.min())
vmaxs.append(ar_grid.max())
vmins.append(0)
vmaxs.append(adj.sum(axis=1).max())
vmins.append(None)
vmaxs.append(None)
data_list.append(ar_grid)
data_list.append(adj.sum(axis=1))
data_list.append('t2m corr')

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
var_name = 'pr'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data2 = ds.ds['anomalies'].data
print(data2.shape)
emp_corr2 = compute_empcorr(data2, 'pearson')#'binMI'
adj2 = get_adj(emp_corr2, density, weighted=False)
degs2 = adj2.sum(axis=1)
vmin2, vmax2 = 0, adj2.sum(axis=1).max()
ar_grid2 = np.diag(autocorr(data2))
vmins.append(ar_grid2.min())
vmaxs.append(ar_grid2.max())
vmins.append(0)
vmaxs.append(adj2.sum(axis=1).max())
vmins.append(None)
vmaxs.append(None)
data_list.append(ar_grid2)
data_list.append(adj2.sum(axis=1))
data_list.append('pr corr')

# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(8)
scale_const=3
ms=40
prefig,preaxs = plt.subplots(2,3, figsize=(scale_const * 3*onefigsize[0],scale_const *2*onefigsize[1]))
xs= np.linspace(ar_grid.min(),ar_grid.max(),200)
ys = (1 + 2 * xs**2 / (1-xs**2)) * 13.5
preaxs[0,2].scatter(ar_grid,degs,s=ms,edgecolors = 'none', alpha=0.4)#s=12
preaxs[0,2].set_xlabel('Lag-1 Autocorrelation')
preaxs[0,2].set_ylabel('Degree')
#preaxs[0,2].plot(xs,ys,linewidth=8,color='tab:orange',label='Equation (1)')
#preaxs[0,2].legend()

preaxs[1,2].scatter(ar_grid2,degs2,s=ms,edgecolors = 'none', alpha=0.4)
preaxs[1,2].set_xlabel('Lag-1 Autocorrelation')
preaxs[1,2].set_ylabel('Degree')
preaxs=enumerate_subplots(preaxs,fontsize=scale_const*16)

plt.savefig(base_path + 'ar_deg_t2m_pr_correlation.png',dpi=300,rasterize=True)
# fix this plot!!!

# %%
adjust_fontsize(3)
import string
def enumerate_subplots(axs, pos_x=0.01, pos_y=0.95, fontsize=16):
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
    #plt.tight_layout()
    return axs

def plot_maps_joint(lon, lat, data_list, plot_type='scatter', central_longitude=0, central_latitude = 0,
            vmins=None, vmaxs=None, colors=None, bar=True,cmap = None,
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = True,scale_const = 3, extends = 'both', norms = None,tickss=None,ms=40,savestring=None):
    
    long_longs = lon
    long_lats = lat
    
    # set projection
    if ctp_projection == 'Mollweide':
        proj = ctp.crs.Mollweide(central_longitude=central_longitude)
    elif ctp_projection == 'PlateCarree':
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
    elif ctp_projection == "Orthographic":
        proj = ctp.crs.Orthographic(central_longitude, central_latitude)
    elif ctp_projection == "EqualEarth":
        proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
    else:
        raise ValueError(f'This projection {ctp_projection} is not available yet!')

    fig,axs = plt.subplots(2,3, subplot_kw={'projection': proj},figsize=(scale_const * 3*onefigsize[0],scale_const *2*onefigsize[1]))
    for data,ax,vmin,vmax,color,label,extend,norm,ticks in zip(data_list,axs.flatten(),vmins,vmaxs,colors,labels,extends,norms,tickss):
        if isinstance(data,str) and data == 't2m corr':
            # plt.rcParams.update(bundles.icml2022())
            # plt.rcParams.update({"figure.dpi": 300})
            xs= np.linspace(ar_grid.min(),ar_grid.max(),200)
            ys = (1 + 2 * xs**2 / (1-xs**2)) * 13.5
            axs[0,2].remove()
            ax = fig.add_subplot(2, 3, 3)
            # adjust_fontsize(8)
            # ax.scatter(ar_grid,degs,s=ms,edgecolors = 'none', alpha=0.4)
            # ax.set_xlabel('Lag-1 Autocorrelation')
            # ax.set_ylabel('Degree')
            # ax.plot(xs,ys,linewidth=8,color='tab:orange',label='Equation (1)')
            # ax.legend()
            ax = preaxs[0,2]
            continue
        elif isinstance(data,str) and data == 'pr corr':
            # plt.rcParams.update(bundles.icml2022())
            # plt.rcParams.update({"figure.dpi": 300})
            axs[1,2].remove()
            ax = fig.add_subplot(2, 3, 6)
            # adjust_fontsize(8)
            # ax.scatter(ar_grid2,degs2,s=ms,edgecolors = 'none', alpha=0.4)
            # ax.set_xlabel('Lag-1 Autocorrelation')
            # ax.set_ylabel('Degree')
            ax = preaxs[1,2]
            continue

        ax.set_global()

        # axes properties
        if earth:
            ax.coastlines()
            ax.add_feature(ctp.feature.BORDERS, linestyle=':')
        if gridlines:
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
        
            
        projection = ctp.crs.PlateCarree(central_longitude=central_longitude)

        # set colormap
        cmap = plt.get_cmap(color)
        
        # plotting
        if plot_type =='scatter':
            if norm is None:
                im = ax.scatter(x=long_longs, y=long_lats,
                                c=data, vmin=vmin, vmax=vmax, cmap=cmap,
                                transform=projection)
            else:
                im = ax.scatter(x=long_longs, y=long_lats,
                                c=data, norm = norm, cmap=cmap,
                                transform=projection)
        elif plot_type == 'colormesh':
            # interpolate grid of points to regular grid
            lon_interp = np.arange(-180,
                                    180,
                                    grid_step)
            lat_interp = np.arange(long_lats.min(),
                                    long_lats.max() + grid_step,
                                    grid_step)

            lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
            new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
            origin_points = np.array([long_longs, long_lats]).T
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            new_values = interp.griddata(origin_points, data, new_points,
                                            method='nearest')
            mesh_values = new_values.reshape(len(lat_interp), len(lon_interp))

            if norm is None:
                im = ax.pcolormesh(
                        lon_mesh, lat_mesh, mesh_values,
                        cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)
            else:
                im = ax.pcolormesh(
                        lon_mesh, lat_mesh, mesh_values,
                        cmap=cmap, norm=norm, transform=projection)
        else:
            raise ValueError("Plot type does not exist!")

        if bar:
            label = ' ' if label is None else label
            if ticks is None:
                cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                    label=label, shrink=0.8, ax=ax)
            else:
                cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                    label=label, shrink=0.8, ax=ax,ticks=ticks)
                cbar.ax.set_xticklabels(ticks)
            cbar.set_label(label=label, size = scale_const*plt.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize = scale_const*plt.rcParams['xtick.labelsize'])
    axs=enumerate_subplots(axs,fontsize=scale_const*16)
    plt.savefig(savestring)
    return fig,axs#{"ax": ax,'fig': fig, "projection": projection}

fig,axs = plot_maps_joint(lon,lat,data_list,scale_const=3,vmins=vmins,vmaxs=vmaxs,colors=colors,labels=labels,extends=extends,norms=norms,tickss=tickss,savestring=base_path +f'joint_ardegplot_new.png')




# %%
dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
var_name = 't2m'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)

# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
degs = adj.sum(axis=1)
vmin, vmax = 0, adj.sum(axis=1).max()

ar_grid = np.diag(autocorr(data))
plt.hist(ar_grid,bins=np.linspace(0.0,0.935,40))
print(ar_grid.max(),ar_grid.min())

from scipy import stats
spearcorr, pval = stats.spearmanr(degs,ar_grid)
print('t2m spearcorr: ', spearcorr)
pear_corr=np.corrcoef(degs,ar_grid)
print('t2m pearson: ', pear_corr)

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
var_name = 'pr'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data2 = ds.ds['anomalies'].data
print(data2.shape)
emp_corr2 = compute_empcorr(data2, 'pearson')#'binMI'
adj2 = get_adj(emp_corr2, density, weighted=False)
vmin2, vmax2 = 0, adj2.sum(axis=1).max()
ar_grid2 = np.diag(autocorr(data2))

plt.hist(ar_grid2,bins=np.linspace(0.0,0.935,40))
print(ar_grid2.max(),ar_grid2.min())

degs2 = adj2.sum(axis=1)
from scipy import stats
spearcorr2, pval2 = stats.spearmanr(degs2,ar_grid2)
print('pr spearcorr: ',spearcorr2)
pear_corr2=np.corrcoef(degs2,ar_grid2)
print('pr pearson: ', pear_corr2)

# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(2)

xs= np.linspace(ar_grid.min(),ar_grid.max(),200)
ys = (1 + 2 * xs**2 / (1-xs**2)) * 13.5

fig,axs = plt.subplots(1,2,figsize=(2*onefigsize[0],1*onefigsize[1]))
axs[0].scatter(ar_grid,degs,s=12,edgecolors = 'none', alpha=0.4)
axs[0].set_xlabel('Lag-1 Autocorrelation')
axs[0].set_ylabel('Degree')
#axs[0].plot(xs,ys,color='tab:orange',label='Equation (1)')
#axs[0].legend()

xs2= np.linspace(ar_grid2.min(),ar_grid2.max(),200)
ys2 = (1 + 2 * xs2**2 / (1-xs2**2))
axs[1].scatter(ar_grid2,degs2,s=12,edgecolors = 'none', alpha=0.4)
axs[1].set_xlabel('Lag-1 Autocorrelation')
axs[1].set_ylabel('Degree')
#axs[1].plot(xs2,ys2,color='tab:orange')


# %%
for var_name in var_names:
    if var_name not in ['pr']:
        continue
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

    mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
    ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
    data = ds.ds['anomalies'].data
    print(data.shape)
    
    # empcorr = np.corrcoef(data.T)
    # max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
    emp_corr = compute_empcorr(data, 'pearson')#'binMI'
    adj = get_adj(emp_corr, density, weighted=False)
    vmin, vmax = 0, adj.sum(axis=1).max()
    # num_falselinks = (dists[adj!=0] > max_dist).sum()
    # thisname = f'{var_name}_pearson_dens{density}.txt'

    # adjust_fontsize(2)
    # plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max')
    # plt.savefig(base_path+f'degree_map_{var_name}_pearson_{density}.png')
    # plt.clf()
    # adjust_fontsize(3)
    # plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max')
    # plt.savefig(base_path+f'degree_map3_{var_name}_pearson_{density}.png')
    # plt.clf()

    ar_grid = np.diag(autocorr(data))
    import matplotlib
    import matplotlib.colors as cl

    # bounds = np.linspace(-1,1,100)
    # norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
    #fixed_cmap = shiftedColorMap(matplotlib.cm.RdBu_r, start=-1, midpoint=0, stop=1, name='fixed')
    vmin = 0.75
    adjust_fontsize(2)
    arplt = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='Reds',vmin=vmin,vmax=ar_grid.max(), ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='both',ticks=np.linspace(-1,1,9))# color='RdBu_r',extend='neither', norm=norm
    plt.savefig(base_path+f'ar_map_{var_name}_min{vmin}.png')
    plt.clf()
    adjust_fontsize(3)
    arplt = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='Reds',vmin=vmin,vmax=ar_grid.max(),ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='both',ticks=np.linspace(-1,1,9))
    plt.savefig(base_path+f'ar_map3_{var_name}_min{vmin}.png')
    plt.clf()

    # adjust_fontsize(2)
    # edge_plot = plot_deg_with_edges(ds.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    # edge_plot['ax'].set_rasterization_zorder(0)
    # plt.savefig(base_path+f'rasterlinks_'+ thisname[:-4]+'.pdf',rasterize = True)
    # edge_plot = plot_deg_with_edges(ds.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    # plt.savefig(base_path+f'rasterlinkspng_'+ thisname[:-4]+'.png')
    # adjust_fontsize(3)
    # edge_plot = plot_deg_with_edges(ds.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    # edge_plot['ax'].set_rasterization_zorder(0)
    # plt.savefig(base_path+f'rasterlinks3_'+ thisname[:-4]+'.pdf',rasterize = True)
    # edge_plot = plot_deg_with_edges(ds.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    # plt.savefig(base_path+f'rasterlinkspng3_'+ thisname[:-4]+'.png')

# %%
# joint ar and deg for t2m and pr
import matplotlib
import matplotlib.colors as cl

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
var_name = 't2m'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)

# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
vmin, vmax = 0, adj.sum(axis=1).max()

# num_falselinks = (dists[adj!=0] > max_dist).sum()
# thisname = f'{var_name}_pearson_dens{density}.txt'
adjust_fontsize(2)
fig,axs = plt.subplots(2,2,figsize=(2*onefigsize[0],2*onefigsize[1]))
deg_map = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max',ax=axs[0,1])
axs[0,1] = deg_map['ax']

ar_grid = np.diag(autocorr(data))
bounds = np.linspace(-1,1,100)
norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
#fixed_cmap = shiftedColorMap(matplotlib.cm.RdBu_r, start=-1, midpoint=0, stop=1, name='fixed')
adjust_fontsize(2)
arplt = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='RdBu_r', norm=norm,vmin=ar_grid.min(),vmax=ar_grid.max(), ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='neither',ticks=np.linspace(-1,1,9),ax=axs[0,0])
axs[0,0] = arplt['ax']


dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
var_name = 'pr'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)

# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
vmin, vmax = 0, adj.sum(axis=1).max()
# num_falselinks = (dists[adj!=0] > max_dist).sum()
# thisname = f'{var_name}_pearson_dens{density}.txt'

deg_map = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max',ax=axs[1,1])
axs[1,1] = deg_map['ax']

ar_grid = np.diag(autocorr(data))
bounds = np.linspace(-1,1,100)
norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
#fixed_cmap = shiftedColorMap(matplotlib.cm.RdBu_r, start=-1, midpoint=0, stop=1, name='fixed')
adjust_fontsize(2)
arplt = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='RdBu_r', norm=norm,vmin=ar_grid.min(),vmax=ar_grid.max(), ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='neither',ticks=np.linspace(-1,1,9),ax=axs[1,0])
axs[1,0] = arplt['ax']

axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path +f'joint_ardegplot.pdf', rasterize=True)

# %%
import matplotlib.gridspec as gridspec
import matplotlib.colors as cl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
adjust_fontsize(2)
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
        pos_x = [pos_x] * len(axs)
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs)
    for n, ax in enumerate(axs):
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


fig = plt.figure()
gs = fig.add_gridspec(2, 2)

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
var_name = 't2m'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)

# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
vmin, vmax = 0, adj.sum(axis=1).max()
ax1 = fig.add_subplot(gs[0, 0])#, projection=ccrs.PlateCarree())
plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='RdBu_r', norm=norm,vmin=ar_grid.min(),vmax=ar_grid.max(), ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='neither',ticks=np.linspace(-1,1,9),ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max',ax=ax2)

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
var_name = 'pr'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)

# empcorr = np.corrcoef(data.T)
# max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)
vmin, vmax = 0, adj.sum(axis=1).max()
# num_falselinks = (dists[adj!=0] > max_dist).sum()
# thisname = f'{var_name}_pearson_dens{density}.txt'

ax4 = fig.add_subplot(gs[1, 1])
plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],adj.sum(axis=1),color='Reds', vmin=vmin,vmax=vmax, ctp_projection='EqualEarth',label='Degree',earth = True,extend='max',ax=ax4)

ar_grid = np.diag(autocorr(data))
bounds = np.linspace(-1,1,100)
norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
#fixed_cmap = shiftedColorMap(matplotlib.cm.RdBu_r, start=-1, midpoint=0, stop=1, name='fixed')
adjust_fontsize(2)
ax3 = fig.add_subplot(gs[1, 0])
plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],ar_grid,color='RdBu_r', norm=norm,vmin=ar_grid.min(),vmax=ar_grid.max(), ctp_projection='EqualEarth',label='Autocorrelation',earth = True,extend='neither',ticks=np.linspace(-1,1,9),ax=ax3)

axs = fig.get_axes()
axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path +f'joint_ardegplot.pdf')

# %%
adjust_fontsize(2)
bounds = np.linspace(-1,1,100)
norm = cl.BoundaryNorm(boundaries=bounds, ncolors=256)
norms = [norm,None,norm,None]

data_list=[]
vmins,vmaxs=[],[]
extends = ['neither','max','neither','max']
labels = ['Autocorrelation','Degree','Autocorrelation','Degree']
colors=['RdBu_r','Reds','RdBu_r','Reds']
tickss=[np.linspace(-1,1,9),None,np.linspace(-1,1,9),None]

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/2m_temperature/era5_singlelevel_monthly_temp2m_1979-2020.nc"
var_name = 't2m'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data = ds.ds['anomalies'].data
print(data.shape)
emp_corr = compute_empcorr(data, 'pearson')#'binMI'
adj = get_adj(emp_corr, density, weighted=False)

ar_grid = np.diag(autocorr(data))

vmins.append(ar_grid.min())
vmaxs.append(ar_grid.max())
vmins.append(0)
vmaxs.append(adj.sum(axis=1).max())
data_list.append(ar_grid)
data_list.append(adj.sum(axis=1))

dataset_nc = f"/mnt/qb/goswami/data/era5/single_pressure_level_monthly/total_precipitation/total_precipitation_sfc_1979_2020.nc"
var_name = 'pr'
mydataset_nc = base_path + f'data_{grid_type}{grid_step}_month_detrended_'+ dataset_nc.split('/',20)[-1]
ds = AnomalyDataset(load_nc=mydataset_nc, detrend=False,grid_step = 5,grid_type='fekete')
data2 = ds.ds['anomalies'].data
print(data2.shape)
emp_corr2 = compute_empcorr(data2, 'pearson')#'binMI'
adj2 = get_adj(emp_corr2, density, weighted=False)
vmin2, vmax2 = 0, adj2.sum(axis=1).max()
ar_grid2 = np.diag(autocorr(data2))
vmins.append(ar_grid2.min())
vmaxs.append(ar_grid2.max())
vmins.append(0)
vmaxs.append(adj2.sum(axis=1).max())
data_list.append(ar_grid2)
data_list.append(adj2.sum(axis=1))

# %%
import string
def enumerate_subplots(axs, pos_x=0.01, pos_y=0.95, fontsize=16):
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
    #plt.tight_layout()
    return axs

def plot_maps_joint(lon, lat, data_list, plot_type='scatter', central_longitude=0, central_latitude = 0,
            vmins=None, vmaxs=None, colors=None, bar=True,cmap = None,
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = True,scale_const = 3, extends = 'both', norms = None,tickss=None,savestring=None):
    
    long_longs = lon
    long_lats = lat
    
    # set projection
    if ctp_projection == 'Mollweide':
        proj = ctp.crs.Mollweide(central_longitude=central_longitude)
    elif ctp_projection == 'PlateCarree':
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
    elif ctp_projection == "Orthographic":
        proj = ctp.crs.Orthographic(central_longitude, central_latitude)
    elif ctp_projection == "EqualEarth":
        proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
    else:
        raise ValueError(f'This projection {ctp_projection} is not available yet!')

    fig,axs = plt.subplots(2,2, subplot_kw={'projection': proj},figsize=(scale_const * 2*onefigsize[0],scale_const *2*onefigsize[1]))
    for data,ax,vmin,vmax,color,label,extend,norm,ticks in zip(data_list,axs.flatten(),vmins,vmaxs,colors,labels,extends,norms,tickss):
        ax.set_global()

        # axes properties
        if earth:
            ax.coastlines()
            ax.add_feature(ctp.feature.BORDERS, linestyle=':')
        if gridlines:
            gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
        
            
        projection = ctp.crs.PlateCarree(central_longitude=central_longitude)

        # set colormap
        cmap = plt.get_cmap(color)
        
        # plotting
        if plot_type =='scatter':
            if norm is None:
                im = ax.scatter(x=long_longs, y=long_lats,
                                c=data, vmin=vmin, vmax=vmax, cmap=cmap,
                                transform=projection)
            else:
                im = ax.scatter(x=long_longs, y=long_lats,
                                c=data, norm = norm, cmap=cmap,
                                transform=projection)
        elif plot_type == 'colormesh':
            # interpolate grid of points to regular grid
            lon_interp = np.arange(-180,
                                    180,
                                    grid_step)
            lat_interp = np.arange(long_lats.min(),
                                    long_lats.max() + grid_step,
                                    grid_step)

            lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
            new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
            origin_points = np.array([long_longs, long_lats]).T
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            new_values = interp.griddata(origin_points, data, new_points,
                                            method='nearest')
            mesh_values = new_values.reshape(len(lat_interp), len(lon_interp))

            if norm is None:
                im = ax.pcolormesh(
                        lon_mesh, lat_mesh, mesh_values,
                        cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)
            else:
                im = ax.pcolormesh(
                        lon_mesh, lat_mesh, mesh_values,
                        cmap=cmap, norm=norm, transform=projection)
        else:
            raise ValueError("Plot type does not exist!")

        if bar:
            label = ' ' if label is None else label
            if ticks is None:
                cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                    label=label, shrink=0.8, ax=ax)
            else:
                cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                    label=label, shrink=0.8, ax=ax,ticks=ticks)
                cbar.ax.set_xticklabels(ticks)
            cbar.set_label(label=label, size = scale_const*plt.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize = scale_const*plt.rcParams['xtick.labelsize'])
    axs=enumerate_subplots(axs,fontsize=scale_const*16)
    plt.savefig(savestring)
    return fig,axs#{"ax": ax,'fig': fig, "projection": projection}

fig,axs = plot_maps_joint(lon,lat,data_list,vmins=vmins,vmaxs=vmaxs,colors=colors,labels=labels,extends=extends,norms=norms,tickss=tickss,savestring=base_path +f'joint_ardegplot.pdf')


# %%

# plot estimation variance scaling as a function of autocorr, as predicted by (1):

# arbins = np.linspace(0,0.9,101)
# variance3d = np.zeros((len(arbins),len(arbins)))

# for ibin,ab in enumerate(arbins):
#     for jbin, ab2 in enumerate(arbins):
#         variance3d[ibin,jbin] = 1+ 2 * ab * ab2 / (1-ab * ab2)

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface(z=variance3d.T, x=arbins, y=arbins)])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='Estimation variance', autosize=False,
#                   scene = dict(
#                     xaxis_title='Autocorr.',
#                     yaxis_title='Autocorr.',
#                     zaxis_title='Variance'),
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()

# %%
