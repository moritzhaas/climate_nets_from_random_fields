import os
import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
# import cartopy as ctp
from climnet.dataset import BaseDataset
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import *
from climnet.event_synchronization import event_synchronization_matrix
import time
from collections import Counter
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 150})
def adjust_fontsize(num_cols):
    keys = ['font.size','axes.labelsize','legend.fontsize','xtick.labelsize','ytick.labelsize','axes.titlesize']
    for key in keys:
        plt.rcParams[key] = bundles.icml2022()[key] * num_cols / 2
adjust_fontsize(3)
#plt.rcParams["figure.figsize"] = (8*2,6*2)

# %%
# Grid
#n_time = 100
alpha = 0.05
n_lat = 18 * 2
n_lon = 2 * n_lat
grid_type = 'fekete'
n_time = 100
num_runs = 30
# grid_stretch = 1
rm_outliers = True
# set parameters
grid_step = 5
var_name = 't2m'
grid_type = 'fekete'
num_cpus = 48
#lon_range = [-110, 40]
#lat_range = [20, 85]
save = True
# time_range = ['1980-01-01', '2019-12-31']
time_range = None
tm = None  # 'week'
norm = True
denslist = [0.005, 0.01, 0.05, 0.1, 0.2]
var = 1
nu=1.5
len_scale = 0.1
ar = 0

# distribution
distrib = 'igrf' # normal
# K = 50
# gamma = 2.5 # 2.1
# var = 10
# A = np.array([(i+1) ** (-gamma) for i in range(K+1)])
# va = 0
# for l in range(len(A)):
#     va += A[l] * (2 * l + 1) / (4 * np.pi)
# A *= var/va


base_path = '../../climnet_output/' #'~/Documents/climnet_output/'
#name = f'{distrib}'
#filename = base_path + name + f'_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}.nc'
name = f'matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_{num_runs}runs'

if os.getenv("HOME") == '/Users/moritz':
    dataset_nc = base_path + f"era5_{var_name}_2.5_ds.nc"
else:
    dataset_nc = f"/mnt/qb/goswami/exchange_folder/climate_data/era5_{var_name}_2.5_ds.nc"

# %%
# generate grid
grid_step_lon = 360/ n_lon
grid_step_lat = 180/ n_lat
dist_equator = gdistance((0,0),(0,grid_step_lon))
lon, lat = regular_lon_lat(n_lon,n_lat)
regular_grid = {'lon': lon, 'lat': lat}
lon2, lat2 = regular_grid['lon'], regular_grid['lat']
dist_equator = gdistance((0,0),(0,grid_step_lon))
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
# %%
lsm = xr.open_dataarray('../input/land-sea-mask_era5.nc')
lsmbool = lsm.data
lsmbool[np.isnan(lsmbool)] = 0
# %%
grid_lsm = np.ones(num_points)
for i in range(num_points):
    grid_lsm[i] = int(lsm.sel(lon=lon[i],lat=lat[i],method ='nearest').data)

lon_sea = lon[grid_lsm == 0]
lat_sea = lat[grid_lsm == 0]

# generate data with fitted correlation structure
'''
distrib = 'igrf'
data = np.zeros((n_time,n_lat,n_lon))
rvs = np.random.normal(size = (len(A) ** 2, n_time + 1))
T = isotropic_grf(A, rvs = rvs)
for i in range(len(lat2)):
    la = lat2[i]
    for j in range(len(lon2)):
        lo = lon2[j]
        if distrib == 'igrf':
            data[:,i,j] = (T(la,lo)[1:])
            data[:,i,j] -= data[:,i,j].mean()
        else:
            data[:,i,j] = np.exp(T(la,lo)[1:])
            data[:,i,j] -= data[:,i,j].mean()
da = xr.DataArray(
    data = data,
    dims=['time', 'lat', 'lon'],
    coords=dict(time=np.arange(start_date, np.timedelta64(n_time, 'D'), dtype='datetime64[D]'),
                lat=lat2,
                lon=lon2),
                name=name
                )
#os.remove(filename)
da.to_netcdf(filename)
'''

#ds = BaseDataset(name, data_nc=filename, grid_type=grid_type, grid_step=grid_step)
# %%
density = 0.005
thisname = f'matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_{num_runs}runs_dens{density}'
#data = ds.ds[name].data[:,grid_lsm == 0]
seagrid = {'lon': lon_sea, 'lat': lat_sea}
# %%
# emp_cov = np.cov(data.reshape((n_time,-1)).T) #ds.ds[name].data.T
# empcorr = np.corrcoef(data.reshape((n_time,-1)).T)
# # emp_cov = np.cov(data.T)
# spempcorr, pval = stats.spearmanr(data.reshape((n_time,-1)))
# emp_mi = np.zeros((data.shape[1], data.shape[1]))
# emp_hsic = np.zeros((data.shape[1], data.shape[1]))
# for i in range(data.shape[1]):
#     for j in range(i):
#         emp_mi[i,j] = revised_mi(data[:,i], data[:,j], q = 2)
#         emp_mi[j,i] = emp_mi[i,j]
#         hsic = HSIC(data[:,i], data[:,j],unbiased=False, permute = False)
#         emp_hsic[i,j] = hsic
#         emp_hsic[j,i] = emp_hsic[i,j]
# sorted_dat = np.sort(np.abs(data),axis=0)
# dat_quantiles = sorted_dat[int(np.ceil(alpha * len(sorted_dat)-1)),:]
# events= (np.abs(data).T >= np.repeat(dat_quantiles.reshape((-1,1)),n_time, axis = 1))
# es = event_synchronization_matrix(events)

# # %%
# def plot_deg_with_edges(grid, adj, transform = ctp.crs.PlateCarree(), ctp_projection='EqualEarth', color ='Reds', pts = None, *args):
#     deg_plot = plot_map_lonlat(grid['lon'],grid['lat'], adj.sum(axis = 0), color=color, ctp_projection=ctp_projection, grid_step=grid_step_lon, *args)
#     if pts is None:
#         edges = np.where(adj != 0)
#         for i in range(len(edges[0])):
#             deg_plot['ax'].plot([grid['lon'][edges[0][i]],grid['lon'][edges[1][i]]], [grid['lat'][edges[0][i]], grid['lat'][edges[1][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
#     else:
#         for point in pts:
#             edges = np.where(adj[point,:] != 0)
#             for i in range(len(edges[0])):
#                 deg_plot['ax'].plot([grid['lon'][point], grid['lon'][edges[0][i]]], [grid['lat'][point], grid['lat'][edges[0][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
#     return deg_plot

# adj = get_adj(empcorr, density)
# mip_eigc, mip_cc, mip_betw = get_mips(adj)
# mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
# edge_plot = plot_deg_with_edges(seagrid, adj, transform = ctp.crs.Geodetic())
# edge_plot['ax'].plot(lon_sea[mip_deg],lat_sea[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_cc],lat_sea[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_betw],lat_sea[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
# edge_plot['fig'].legend()
# plt.title('Empirical correlation ' + thisname)
# plt.savefig(base_path+f'links_corr_'+ thisname, dpi = 150)

# adj = get_adj(spempcorr, density)
# mip_eigc, mip_cc, mip_betw = get_mips(adj)
# mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
# edge_plot = plot_deg_with_edges(seagrid,adj, transform = ctp.crs.Geodetic())
# edge_plot['ax'].plot(lon_sea[mip_deg],lat_sea[mip_deg], 'o',color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_cc],lat_sea[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_betw],lat_sea[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
# edge_plot['fig'].legend()
# plt.title('Empirical Spearman correlation ' + thisname)
# plt.savefig(base_path+f'links_spcorr_'+ thisname, dpi = 150)

# adj = get_adj(emp_mi, density)
# mip_eigc, mip_cc, mip_betw = get_mips(adj)
# mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
# edge_plot = plot_deg_with_edges(seagrid, adj, transform = ctp.crs.Geodetic())
# edge_plot['ax'].plot(lon_sea[mip_deg],lat_sea[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_cc],lat_sea[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_betw],lat_sea[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
# edge_plot['fig'].legend()
# plt.title('Empirical MI ' + thisname)
# plt.savefig(base_path+f'links_MI_'+ thisname, dpi = 150)

# adj = get_adj(emp_hsic, density)
# mip_eigc, mip_cc, mip_betw = get_mips(adj)
# mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
# edge_plot = plot_deg_with_edges(seagrid, adj, transform = ctp.crs.Geodetic())
# edge_plot['ax'].plot(lon_sea[mip_deg],lat_sea[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_cc],lat_sea[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_betw],lat_sea[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
# edge_plot['fig'].legend()
# plt.title('Empirical HSIC ' + thisname)
# plt.savefig(base_path+f'links_HSIC_'+ thisname, dpi = 150)

# adj = get_adj(es, density)
# mip_eigc, mip_cc, mip_betw = get_mips(adj)
# mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
# edge_plot = plot_deg_with_edges(seagrid, adj, transform = ctp.crs.Geodetic())
# edge_plot['ax'].plot(lon_sea[mip_deg],lat_sea[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_cc],lat_sea[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
# edge_plot['ax'].plot(lon_sea[mip_betw],lat_sea[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
# edge_plot['fig'].legend()
# plt.title('Empirical ES ' + thisname)
# plt.savefig(base_path+f'links_ES_'+ thisname, dpi = 150)

# %%

# distribution
from sklearn.gaussian_process.kernels import Matern
denslist = [0.005, 0.05, 0.1, 0.2]
filter_string = f'*matern_nu{nu}_len{len_scale}_ar0_fekete36_time100_*'
# K = 50
# gamma = 2.5 # 2.1
# var = 10
# num_runs = 50
# A = np.array([(i+1) ** (-gamma) for i in range(K+1)])
degs = np.zeros((num_runs,len(denslist),len(lon_sea)))
ccs = np.zeros((num_runs,len(denslist),len(lon_sea)))
betws = np.zeros((num_runs,len(denslist),len(lon_sea)))
# do multiple runs and measure degree density on locations
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

max_runs = len(find(filter_string, base_path+ 'empdata/'))
if max_runs != 30:
    print('max_runs not 30 but ', max_runs)
# %%
for irun in range(num_runs):
    if max_runs <= irun:
        seed = int(time.time())
        np.random.seed(seed)
        data = diag_var_process(np.zeros(num_points), cov, n_time)
        mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar0_fekete{n_lat}_time{n_time}_var1_seed{seed}.txt',data)
    else:
        data = myload(find(filter_string, base_path+ 'empdata/')[irun])
    for j in range(data.shape[1]):
        data[:,j] -= data[:,j].mean()
        data[:,j] /= data[:,j].std()
    
    empcorr = compute_empcorr(data, 'pearson')[grid_lsm == 0,:][:,grid_lsm == 0]
    for i,dens in enumerate(denslist):
        adj = get_adj(empcorr, dens, weighted = False)
        degs[irun,i,:] = adj.sum(axis=0)
        G = nx.from_numpy_matrix(adj)
        betws[irun,i,:] = list(nx.betweenness_centrality(G).values())
        ccs[irun,i,:] = list(nx.clustering(G).values())
    # adj = get_adj(empcorr, 0.05, weighted = False)
    # G = nx.from_numpy_matrix(adj)
    # ccs[irun,:] = list(nx.clustering(G).values())
    # adj = get_adj(empcorr, 0.2, weighted = False)
    # degs[irun,:] = adj.sum(axis=0)

landsea_exper = [degs, betws,ccs]
mysave(base_path, f'landsea_exper_nu{nu}_len{len_scale}_{num_runs}.txt',landsea_exper)

# %%
# dens = 0.2
# plot_map_lonlat(lon_sea,lat_sea, degs.mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True, grid_step=grid_step, label = f'Avg. degree, dens={dens}')
# plt.savefig(base_path+ f'deg_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
# dens=0.05
# plot_map_lonlat(lon_sea,lat_sea, ccs.mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True, grid_step=grid_step, label = f'Avg. clustering coefficient, dens={dens}')
# plt.savefig(base_path+ f'cc_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
# dens = 0.005
# plot_map_lonlat(lon_sea,lat_sea, betws.mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True, grid_step=grid_step, label = f'Avg. betweenness, dens={dens}')
# plt.savefig(base_path+ f'betw_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
# %%
for i,dens in enumerate(denslist):
    plot_map_lonlat(lon_sea,lat_sea, degs[:,i,:].mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Avg. degree, dens={dens}')
    plt.savefig(base_path+ f'deg_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
    plt.clf()
    plot_map_lonlat(lon_sea,lat_sea, ccs[:,i,:].mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Avg. clustering coefficient, dens={dens}')
    plt.savefig(base_path+ f'cc_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
    plt.clf()
    plot_map_lonlat(lon_sea,lat_sea, betws[:,i,:].mean(axis=0), color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Avg. betweenness, dens={dens}')
    plt.savefig(base_path+ f'betw_mean_dens{dens}_nu{nu}_len{len_scale}_{num_runs}.pdf')
    plt.clf()
# %%
for i,dens in enumerate(denslist):
    plot_map_lonlat(lon_sea,lat_sea, degs[:,i,:].std(axis=0), color='Reds', ctp_projection='EqualEarth', grid_step=grid_step, label = f'Std degree, dens={dens}')
    plt.savefig(base_path+ f'deg_std_dens{dens}.pdf')
    plt.clf()
    plot_map_lonlat(lon_sea,lat_sea, ccs[:,i,:].std(axis=0), color='Reds', ctp_projection='EqualEarth', grid_step=grid_step, label = f'Std cc, dens={dens}')
    plt.savefig(base_path+ f'cc_std_dens{dens}.pdf')
    plt.clf()
    plot_map_lonlat(lon_sea,lat_sea, betws[:,i,:].std(axis=0), color='Reds', ctp_projection='EqualEarth', grid_step=grid_step, label = f'Std betw, dens={dens}')
    plt.savefig(base_path+ f'betw_std_dens{dens}.pdf')
    plt.clf()
# %%
degs2,betws2, ccs2 = [np.zeros((len(denslist),len(lon_sea))) for _ in range(3)]
for i,dens in enumerate(denslist):
    adj = get_adj(cov[grid_lsm == 0,:][:,grid_lsm == 0], dens, weighted = False)
    degs2[i,:] = adj.sum(axis=0)
    G = nx.from_numpy_matrix(adj)
    betws2[i,:] = list(nx.betweenness_centrality(G).values())
    ccs2[i,:] = list(nx.clustering(G).values())

# %%
landsea_exper2 = [degs2, betws2,ccs2]
mysave(base_path, f'landsea_exper_true_nu{nu}_len{len_scale}_{num_runs}.txt',landsea_exper2)

# for i,dens in enumerate(denslist):
#     plot_map_lonlat(lon_sea,lat_sea, degs2[i,:], color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Degree, dens={dens}')
#     plt.savefig(base_path+ f'deg_dens{dens}_true_nu{nu}_len{len_scale}_{num_runs}.pdf')
#     plt.clf()
#     plot_map_lonlat(lon_sea,lat_sea, ccs2[i,:], color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Clustering coefficient, dens={dens}')
#     plt.savefig(base_path+ f'cc_dens{dens}_true_nu{nu}_len{len_scale}_{num_runs}.pdf')
#     plt.clf()
#     plot_map_lonlat(lon_sea,lat_sea, betws2[i,:], color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Betweenness, dens={dens}')
#     plt.savefig(base_path+ f'betw_dens{dens}_true_nu{nu}_len{len_scale}_{num_runs}.pdf')
#     plt.clf()

# %%
# joint plot for paper
degs2, betws2,ccs2 = myload(base_path + f'landsea_exper_true_nu{nu}_len{len_scale}_{num_runs}.txt')

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
            vmin=None, vmax=None, color='Reds', bar=True,cmap = None,
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = True,scale_const = 3, extend = 'both', norm = None,ticks=None,savestring=None):
    
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

    fig,axs = plt.subplots(1,3, subplot_kw={'projection': proj},figsize=(scale_const * 3*onefigsize[0],scale_const *onefigsize[1]))
    for data,ax,label in zip(data_list,axs.flatten(),labels):
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


data_list = [degs2[-1,:],ccs2[2,:], betws2[0,:]]
labels = [f'Degree, dens={denslist[-1]}', f'Clustering coefficient, dens={denslist[2]}', f'Betweenness, dens={denslist[0]}']

fig,axs = plot_maps_joint(lon_sea,lat_sea,data_list,labels=labels,savestring=base_path +f'joint_gridbiasplot.pdf')
fig,axs = plot_maps_joint(lon_sea,lat_sea,data_list,labels=labels,savestring=base_path +f'joint_gridbiasplot.png')





# %%
dens = 0.1
adj = get_adj(cov[grid_lsm == 0,:][:,grid_lsm == 0], dens, weighted = False)
#adj = knn_adj(cov[grid_lsm == 0,:][:,grid_lsm == 0], 3, weighted = False)
G = nx.from_numpy_matrix(adj)
thisbetw = list(nx.betweenness_centrality(G).values())
plot_map_lonlat(lon_sea,lat_sea, thisbetw, color='Reds',vmax=0.006, ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Betweenness, dens={dens}')
plt.savefig(base_path+ f'betw_dens{dens}_true_nu{nu}_len{len_scale}.pdf')
# %%
dens = 0.05
adj = get_adj(cov[grid_lsm == 0,:][:,grid_lsm == 0], dens, weighted = False)
#adj = knn_adj(cov[grid_lsm == 0,:][:,grid_lsm == 0], 3, weighted = False)
G = nx.from_numpy_matrix(adj)
thisbetw = list(nx.clustering(G).values())
plot_map_lonlat(lon_sea,lat_sea, thisbetw, color='Reds',vmax=0.8, ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Clustering coefficient, dens={dens}')
plt.savefig(base_path+ f'cc_dens{dens}_true_nu{nu}_len{len_scale}.pdf')

#idx = 803
#plot_map_lonlat(lon_sea,lat_sea, cov[grid_lsm == 0,:][:,grid_lsm == 0][idx,:], color='Reds', ctp_projection='EqualEarth',earth = True,  grid_step=grid_step, label = f'Betweenness, dens={dens}')
