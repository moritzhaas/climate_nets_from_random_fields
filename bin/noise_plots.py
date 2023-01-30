# %%
import networkx as nx
import os
import numpy as np
from numpy.random import f
import scipy.interpolate as interp
from scipy import stats, ndimage, special
import xarray as xr
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import revised_mi
import time
start_time = time.time()
#plt.style.use('bmh')



num_runs = 3

# Network parameters
n_lat = 18 * 2 #4 # makes resolution of 180 / n_lat degrees
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = True
corr_method='spearman' # 'spearman', 'MI', 'HSIC', 'ES'
ranks = False
robust_tolerance = 0.25
denslist = [0.005,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
ks = [5,  10, 65, 125, 250]
influence = 0 #0.8
corruption = 0.5
if len(denslist) != len(ks) and typ == 'threshold':
    raise RuntimeError('Denslist needs to have same length as ks.')

# data parameters
distrib = 'igrf'
n_time = 100
# K = 50
# gamma = 2.5 # 2.1
# var = 10
# A = np.array([(i+1) ** (-gamma) for i in range(K+1)])

nu = 1.5
len_scale = 0.2

exec(open("grid_helper.py").read())
noisequant = np.zeros_like(all_dens)

    
# diagonal VAR1 coeff of length num_points
ar = 0
ar_coeff = ar * np.ones(num_points)

#rd_idcs = np.random.permutation(np.arange(num_points))[:num_points // 2]
#ar_coeff[rd_idcs] = ac2 * np.ones(len(rd_idcs))

south = np.where(lat < 0)[0]

thisname = f'noise{corruption}_{corr_method}_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1.txt'

def noise_grf(igrf, cleanidcs, corruption = 0):
    ntime, npoints = igrf.shape
    noise = np.random.randn(ntime, npoints)
    noise[:,cleanidcs] = 0
    return igrf + corruption * noise

def plot_deg_with_edges(grid, adj, transform = ctp.crs.PlateCarree(),label = None,  ctp_projection='EqualEarth', color ='Reds', pts = None, *args):
    deg_plot = plot_map_lonlat(lon,lat, adj.sum(axis = 0), color=color,label =label, ctp_projection=ctp_projection, grid_step=grid_step_lon, *args)
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
corruption =0.7
knn = 5
weighted = True
tweighted = False
density = 0.005
thisname = f'noise{corruption}_{corr_method}_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1.txt'
datafiles = find(f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*', base_path+'empdata/')
if datafiles != []:
    data_igrf = myload(datafiles[0])
else:
    cartesian_grid = spherical2cartesian(lon,lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    cov = kernel(cartesian_grid)
    seed = int(time.time())
    np.random.seed(seed)
    data_igrf = diag_var_process(ar_coeff, cov, n_time)
data = noise_grf(data_igrf, cleanidcs=south,corruption=corruption)
emp_corr = compute_empcorr(data, similarity=corr_method)

adj = get_adj(emp_corr, density,weighted=tweighted)
adj2 = knn_adj(emp_corr, knn,weighted=weighted)
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
            vmin=None, vmax=None, color='Reds', bar=True,cmap = None,
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = False,scale_const = 3, extends = 'both', norm= None,ticks=None,savestring=None):
    
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

    fig,axs = plt.subplots(1,2, subplot_kw={'projection': proj},figsize=(scale_const * 2*onefigsize[0],scale_const *onefigsize[1]))
    for adj,ax,label,extend in zip(adj_list,axs.flatten(),labels,extends):
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

adj_list = [adj,adj2]
labels = [f'Degree', f'Degree']
extends = ['both','max']

fig,axs = plot_edgemaps_joint(lon,lat,adj_list,color = 'Reds',labels=labels, extends=extends,savestring=base_path +f'joint_noiselinkplot.pdf')
# %%
fig,axs = plot_edgemaps_joint(lon,lat,adj_list,color = 'Reds',labels=labels, extends=extends,savestring=base_path +f'joint_noiselinkplot.png')



#edge_plot['ax'].plot(lon[mip_deg],lat[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[mip_cc],lat[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[mip_betw],lat[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[teleidcs],lat[teleidcs], 'o', color = 'purple', label = 'tele', transform = ctp.crs.PlateCarree())
#edge_plot['fig'].legend()
#plt.title(f'dens{np.round(adj.sum()/(num_points*(num_points-1)),3)}' + thisname)
# try:
#     plt.savefig(base_path+f'links_thres_w{tweighted}_'+ thisname+'.pdf', dpi = 150)
# except PermissionError:
#     os.remove(base_path+f'links_thres_w{tweighted}_'+ thisname+'.pdf')
#     plt.savefig(base_path+f'links_thres_w{tweighted}_'+ thisname+'.pdf', dpi = 150)


#mip_eigc, mip_cc, mip_betw = get_mips(adj)
#mip_deg = np.argsort(np.sum(adj,axis = 0))[-10:]
#edge_plot['ax'].plot(lon[mip_deg],lat[mip_deg], 'o', color = 'yellow', label = 'mip deg', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[mip_cc],lat[mip_cc], 'go',label = 'mip clust', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[mip_betw],lat[mip_betw], 'bo',label = 'mip between', transform = ctp.crs.PlateCarree())
#edge_plot['ax'].plot(lon[teleidcs],lat[teleidcs], 'o', color = 'purple', label = 'tele', transform = ctp.crs.PlateCarree())
#edge_plot['fig'].legend()
#plt.title(f'dens{np.round(adj.sum()/(num_points*(num_points-1)),3)}' + thisname)
# try:
#     plt.savefig(base_path+f'links_knn_w{weighted}_'+ thisname+'.pdf', dpi = 150)
# except PermissionError:
#     os.remove(base_path+f'links_knn_w{weighted}_'+ thisname+'.pdf')
#     plt.savefig(base_path+f'links_knn_w{weighted}_'+ thisname+'.pdf', dpi = 150)

# %%
plt.hist(dists[adj!=0],bins = 50)
# %%
south
plt.scatter(dists[4,:],emp_corr[4,:],s=1)
# %%
# plot degree distribution of noisy pts and rest

