# %%
import cartopy
import matplotlib
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from climnet.grid import FeketeGrid, regular_lon_lat
from climnet.myutils import *
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 150})
adjust_fontsize(3)

# std parameters
num_bins = 100
n_lat = 2*18
num_runs = 30
n_time = 500
ar = 0
ar2 = None
distrib = 'igrf'
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = False
ranks = False
corr_method='pearson' # 'spearman', 'MI', 'HSIC', 'ES'
robust_tolerance = 0.5
denslist = [0.001,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
ks = [6, 60, 300, 600,1200] #[5,  10, 65, 125, 250]

if len(denslist) != len(ks) and typ == 'threshold':
    raise RuntimeError('Denslist needs to have same length as ks.')

# covariance parameters
var = 10
nu = 1.5
len_scale = 0.2
dens = 0.005


base_path = '../../climnet_output/' #'~/Documents/climnet_output/'

# %%
# generate grid
n_lon = 2 * n_lat
grid_step_lon = 360/ n_lon
grid_step_lat = 180/ n_lat
dist_equator = gdistance((0,0),(0,grid_step_lon))
lon, lat = regular_lon_lat(n_lon,n_lat)
regular_grid = {'lon': lon, 'lat': lat}
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

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99
thisname = f'matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1.pdf'
# %%
curr_time = time.time()
data = np.zeros((n_time,num_points))
data2 = data
cartesian_grid = spherical2cartesian(lon,lat)
degbins = np.linspace(0,0.02, num_bins+1)
llbins = np.linspace(0, 0.3, num_bins+1)
all_lls = np.zeros((num_bins, num_runs))
all_degs = np.zeros((num_bins, num_runs))
true_degs = np.zeros((num_bins))
all_llquant1 = np.zeros((num_runs))
all_llquant2 = np.zeros((num_runs))
all_degs2 = all_degs
from sklearn.gaussian_process.kernels import Matern
# generate data with chordal Matern covariance
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(cartesian_grid)
ar_coeff = np.zeros(len(cov))


# %%
#density = 0.005
seed = 1657551699#int(time.time())
np.random.seed(seed)
data = diag_var_process(ar_coeff, cov, n_time)
data = np.sqrt(var) * data
exp_data = np.exp(np.sqrt(var) * data)
data2 = exp_data
# save with nam in name
for j in range(data.shape[1]):
    data[:,j] -= data[:,j].mean()
    data[:,j] /= data[:,j].std()
    data2[:,j] -= exp_data[:,j].mean()
    data2[:,j] /= exp_data[:,j].std()


emp_corr2 = compute_empcorr(data2, corr_method)
emp_corr = compute_empcorr(data2, 'spearman')
adj = get_adj(emp_corr, dens, weighted=False)
adj2 = get_adj(emp_corr2, dens, weighted=False)

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

adj3 = get_adj(5-dists, dens, weighted = False) # adjust vmax to expigrf
#mip_eigc, mip_cc, mip_betw = get_mips(adj)
#mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]

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

def plot_edgemaps_joint(lon, lat, adj_list, plot_type='scatter', central_longitude=0, central_latitude = 0,
            vmins=None, vmaxs=None, colors=None, bar=True,cmap = None,
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = False,scale_const = 3, extend = 'both', norm= None,ticks=None,savestring=None):
    
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
    for adj,ax,vmin,vmax,color,label in zip(adj_list,axs.flatten(),vmins,vmaxs,colors,labels):
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
                                c=adj.sum(axis = 0), vmin=vmin, vmax=vmax, cmap=cmap,
                                transform=projection)
            else:
                im = ax.scatter(x=long_longs, y=long_lats,
                                c=adj, norm = norm, cmap=cmap,
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
            new_values = interp.griddata(origin_points, adj.sum(axis=0), new_points,
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
        edges = np.where(adj != 0)
        for i in range(len(edges[0])):
            ax.plot([lon[edges[0][i]],lon[edges[1][i]]], [lat[edges[0][i]], lat[edges[1][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=ctp.crs.Geodetic())
    axs=enumerate_subplots(axs,fontsize=scale_const*16)
    plt.savefig(savestring)
    return fig,axs#{"ax": ax,'fig': fig, "projection": projection}

vmins = [0,0,0]
vmax = adj2.sum(axis=1).max()
vmaxs=[vmax,vmax,vmax]
labels = ['Degree','Degree','Degree']
thesecolors= ['Reds','Reds','Reds']
adj_list = [adj3,adj2,adj]
fig,axs = plot_edgemaps_joint(lon,lat,adj_list,vmins=vmins,vmaxs=vmaxs,colors=thesecolors,labels=labels,savestring=base_path+f'jointlinkplots_seed{seed}_'+ thisname)
fig,axs = plot_edgemaps_joint(lon,lat,adj_list,vmins=vmins,vmaxs=vmaxs,colors=thesecolors,labels=labels,savestring=base_path+f'jointlinkplots_seed{seed}_'+ thisname[:-3]+'png')