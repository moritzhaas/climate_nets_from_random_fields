# %%
from copy import deepcopy
from email.mime import base
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

betw_dens = [0.004,0.0049,0.005,0.0051,0.006,0.08,0.098,0.1,0.102,0.12]
betws = np.zeros((len(betw_dens),num_points))
# %%
for var_name in var_names:
    if var_name != 't2mdaily':
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
    break
    # empcorr = np.corrcoef(data.T)
    # max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
    emp_corr = compute_empcorr(data, 'binMI')#'binMI'
    for idiff,thisdens in enumerate(betw_dens):
        adj = get_adj(emp_corr,thisdens,weighted=False)
        vmin, vmax = 0, adj.sum(axis=1).max()
        G = nx.from_numpy_matrix(adj)
        betws[idiff,:] = list(nx.betweenness_centrality(G).values())
        # num_falselinks = (dists[adj!=0] > max_dist).sum()
        # thisname = f'{var_name}_pearson_dens{density}.txt'

        #ar_grid = np.diag(autocorr(data))
        # adjust_fontsize(2)
        # plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],betws[idiff,:],color='RdBu_r',ctp_projection='EqualEarth',label='Betweenness',earth = True)
        # plt.savefig(base_path+f'betw_map_{density}_{thisdens}_{var_name}.pdf')
        # plt.clf()
        # adjust_fontsize(3)
        # plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],betws[idiff,:],color='RdBu_r',ctp_projection='EqualEarth',label='Betweenness',earth = True)
        # plt.savefig(base_path+f'betw_map3_{density}_{thisdens}_{var_name}.pdf')
        # plt.clf()

#mysave(base_path,f'realbetws_dens_t2mdaily.txt',betws)

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
import seaborn as sns
cols = sns.color_palette("Spectral", as_cmap=True).reversed()#'RdBu_r'
cdict = cols.__dict__['_segmentdata']
for ke, val in cdict.items():
    for i in range(len(val)):
        val[i] = (0.2+ 0.8 * val[i][0], val[i][1], val[i][2])
    val.reverse()
    val.append((0,1,1))
    val.reverse()
cols.__dict__['_segmentdata'] = cdict
import matplotlib.colors as colors
newcols = colors.LinearSegmentedColormap('Donges',cdict)
vmin,vmax = 0,0.05 # dynamic, see below
betws=myload(base_path+f'realbetws_dens_t2mdaily.txt')
num_points = betws.shape[1]
var_name = 't2mdaily'
adjust_fontsize(3)
#fig, axs = plt.subplots(2,3,figsize=(3*onefigsize[0],2*onefigsize[1]))
#iax = 0
vmaxs,vmins=[],[]
data_list= []
thesecolors = [newcols for _ in range(6)]
labels = ['Betweenness' for _ in range(6)]
for idens in [0,2,4,5,7,9]:
    dens = betw_dens[idens]
    #for idens,dens in enumerate(betw_dens):
    #ar_grid = np.diag(autocorr(data))
    thesebetw = np.log10(1+ (num_points-1) * (num_points -2) * betws[idens,:] / 2)
    vmin = thesebetw.min() # 0 #np.log10(1+betws[idens,:]).min()
    if dens <=0.006:
        vmax = thesebetw.max() #np.log10(1+betws[idens,:]).max() / 2 # 0.025
    else:
        vmax = thesebetw.max()
    vmaxs.append(vmax)
    vmins.append(vmin)
    data_list.append(thesebetw)
    #adjust_fontsize(2)
    #betpl = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = True)#'RdBu_r'
    
    #plt.savefig(base_path+f'betw_map_log_{dens}_{var_name}.pdf')
    #plt.clf()
    
#     betpl = plot_map_lonlat(ds.grid['lon'],ds.grid['lat'],thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = True)
#     ax = betpl['ax']
#     #iax += 1
# plt.savefig(base_path+f'jointbetw_map3_log_{var_name}.pdf')
    #plt.clf()

#mysave(base_path,f'realbetws_t2mdaily.txt',betws)

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
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = True,scale_const = 3, extend = 'both', norm= None,ticks=None,savestring=None):
    
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
    for data,ax,vmin,vmax,color,label in zip(data_list,axs.flatten(),vmins,vmaxs,colors,labels):
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

fig,axs = plot_maps_joint(lon,lat,data_list,vmins=vmins,vmaxs=vmaxs,colors=thesecolors,labels=labels,savestring=base_path+f'jointbetw_map3_log_{var_name}.pdf')


# %%
