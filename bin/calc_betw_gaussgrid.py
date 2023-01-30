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
import time
#import seaborn as sn
#from GraphRicciCurvature.OllivierRicci import OllivierRicci
#from GraphRicciCurvature.FormanRicci import FormanRicci
from climnet.dataset_new import AnomalyDataset
from climnet.similarity_measures import *
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()
curr_time = time.time()

from tueplots import bundles, figsizes
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})

print('HOME is: ', os.getenv("HOME"))
distrib = 'igrf'

grid_type = 'fekete' #'fekete'
n_lat = 18*4
typ ='threshold' # iterated over
corr_method='pearson'
weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True

denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
# ks = [6, 60, 300, 600,1200]

#filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'


#grid_helper and calc_true
num_runs = 5
n_time = 500
nu = 1.5
len_scale = 0.2

# %%
#formanbins = np.linspace(-200,100,301)
n_estim = 500
n_pergroup = 750
ps = [0.005,0.01,0.05]
frac = 0.1
num_bins = 100

betwbins = np.linspace(0,0.04, num_bins + 1)
all_betw = np.zeros((len(ps), num_bins, num_runs))
all_betw4 = np.zeros((len(ps), num_bins, num_runs))
#all_forman = np.zeros((len(ps), len(formanbins)-1, num_runs))
#all_forman4 = np.zeros((len(ps), len(formanbins)-1, num_runs))

density = np.zeros((len(ps),num_runs))
base_path = '../../climnet_output/' #'~/Documents/climnet_output/'

#betw_dens = [0.004,0.0049,0.005,0.0051,0.006,0.08,0.098,0.1,0.102,0.12]
dens=0.005

filter_string = f'*matern_nu{nu}_len{len_scale}_ar0_regular{n_lat}_time{n_time}_*'
similarity = 'pearson'
betwbins2 = np.linspace(0,0.075, num_bins + 1)

exec(open("grid_helper.py").read())
reg_dists, lon2, lat2 = reg_dists, lon2, lat2

from sklearn.gaussian_process.kernels import Matern
# take Gaussian grid and generate data several times
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
lon2,lat2 = np.meshgrid(lon2,lat2)
lon2,lat2 = lon2.flatten(), lat2.flatten()
cov = kernel(spherical2cartesian(lon2,lat2))
num_points = len(lat2)
betws = np.zeros(num_points) # np.zeros((len(betw_dens),num_points))

# %%
max_runs = len(find(filter_string, base_path+ 'empdata/'))
if max_runs != 30:
    print('max_runs not 30 but ', max_runs)
all_betw2 = np.zeros((len(denslist), num_bins, num_runs))
#all_forman2 = np.zeros((len(denslist), len(formanbins)-1,num_runs))
density2 = np.zeros((len(denslist),num_runs))
for irun in range(num_runs):
    if max_runs <= irun:
        seed = int(time.time())+irun
        np.random.seed(seed)
        data = diag_var_process(np.zeros(num_points), cov, n_time)
        mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar0_regular72_time{n_time}_var1_seed{seed}.txt',data)
    else:
        data = myload(find(filter_string, base_path+ 'empdata/')[irun])
        seed = np.int64(find(filter_string, base_path+ 'empdata/')[irun].split('seed',1)[1].split('.txt',1)[0])
    for j in range(data.shape[1]):
        data[:,j] -= data[:,j].mean()
        data[:,j] /= data[:,j].std()
    
    emp_corr = compute_empcorr(data, corr_method)
    #for i, dens in enumerate(betw_dens):
    adj = get_adj(emp_corr,dens, weighted = False)
    deg = adj.sum(axis = 0)/(adj.shape[0]-1)
    G = nx.from_numpy_matrix(adj)
    density2[irun] = nx.density(G) #[i,irun]
    betws = list(nx.betweenness_centrality(G).values())#[i,:]
    # save and plot betweenness
    mysave(base_path,f'betws_dens{dens}_'+filter_string[1:-2]+f'_seed{seed}.txt',[betws,density2])

# for irun in range(num_runs):
#     if max_runs <= irun:
#         seed = int(time.time())
#         np.random.seed(seed)
#         data = diag_var_process(np.zeros(num_points), cov, n_time)
#         mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar0_regular72_time{n_time}_var1_seed{seed}.txt',data)
#     else:
#         data = myload(find(filter_string, base_path+ 'empdata/')[irun])
#         seed = np.int64(find(filter_string, base_path+ 'empdata/')[irun].split('seed',1)[1].split('.txt',1)[0])
#     for j in range(data.shape[1]):
#         data[:,j] -= data[:,j].mean()
#         data[:,j] /= data[:,j].std()
    
#     emp_corr = compute_empcorr(data, corr_method)
#     for i, dens in enumerate(betw_dens):
#         adj = get_adj(emp_corr,dens, weighted = False)
#         deg = adj.sum(axis = 0)/(adj.shape[0]-1)
#         G = nx.from_numpy_matrix(adj)
#         rc = FormanRicci(G, verbose="INFO")
#         rc.compute_ricci_curvature()
#         all_forman2[i,:,irun] = np.histogram(list(nx.get_edge_attributes(rc.G, 'formanCurvature').values()),bins=formanbins)[0]

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

num_points = len(lat2)
betw_files = find(f'betws_dens{dens}_*time{n_time}*', base_path)

data_list = []
vmins,vmaxs= [],[]
for iax in range(3):
    if n_time == 500:
        if iax == 0:
            file = find(f'betws_dens{dens}_*seed1665667991*', base_path)[0]
        elif iax == 1:
            file = find(f'betws_dens{dens}_*seed1665659237*', base_path)[0]
        elif iax == 2:
            file = find(f'betws_dens{dens}_*seed1665650781*', base_path)[0]
    elif n_time == 100:
        if iax == 0:
            file = find(f'betws_dens{dens}_*seed1654881508*', base_path)[0]
        elif iax == 1:
            file = find(f'betws_dens{dens}_*seed1654892050*', base_path)[0]
        elif iax == 2:
            file = find(f'betws_dens{dens}_*seed1654897329*', base_path)[0]
    savename = file.split('/',10)[-1][:-4]
    betws,density2 = myload(file)
    betws = np.array(betws)
    if len(betws) != num_points:
        print(savename + f'has {len(betws)} points.')
        continue    
    vmin,vmax = 0,0.05 # dynamic, see below

    thesebetw = np.log10(1+ (num_points-1) * (num_points -2) * betws / 2)
    vmin = thesebetw.min() # 0 #np.log10(1+betws[idens,:]).min()
    if dens <=0.006:
        vmax = thesebetw.max() #np.log10(1+betws[idens,:]).max() / 2 # 0.025
    else:
        vmax = thesebetw.max()
    data_list.append(thesebetw)
    vmins.append(vmin)
    vmaxs.append(vmax)
thesecolors = [newcols for _ in range(3)]
labels=['Betweenness' for _ in range(3)]

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
            ax=None, ctp_projection="EqualEarth", labels= None, grid_step=2.5, gridlines = True, earth = False,scale_const = 3, extend = 'both', norm = None,ticks=None,savestring=None):
    
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

fig,axs = plot_maps_joint(lon2,lat2,data_list,vmins=vmins,vmaxs=vmaxs,colors=thesecolors,labels=labels,savestring=base_path +f'joint_betws_gaussgrid_time{n_time}.png')


# %%
# expect spurious paths of important betweenness around the equator: poles strongly connected because grid points close, density chosen such that other links barely form
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

num_points = len(lat2)
betw_files = find(f'betws_dens{dens}_*time{n_time}*', base_path)

for irun in range(len(betw_files)):
    savename = betw_files[irun].split('/',10)[-1][:-4]
    betws,density2 = myload(betw_files[irun])
    betws = np.array(betws)
    if len(betws) != num_points:
        print(savename + f'has {len(betws)} points.')
        continue    
    vmin,vmax = 0,0.05 # dynamic, see below

    thesebetw = np.log10(1+ (num_points-1) * (num_points -2) * betws / 2)
    vmin = thesebetw.min() # 0 #np.log10(1+betws[idens,:]).min()
    if dens <=0.006:
        vmax = thesebetw.max() #np.log10(1+betws[idens,:]).max() / 2 # 0.025
    else:
        vmax = thesebetw.max()
    adjust_fontsize(2)
    betpl = plot_map_lonlat(lon2,lat2,thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = False)#'RdBu_r'
    plt.savefig(base_path+f'map_' + savename + '.pdf')
    plt.clf()
    adjust_fontsize(3)
    betpl = plot_map_lonlat(lon2,lat2,thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = False)#'RdBu_r'
    plt.savefig(base_path+ f'map_' + savename + '_size3.pdf')
    plt.clf()

# %%
betw_files = [f for f in find(f'betws_dens{dens}_*nu1.5*len0.2*regular*', base_path) if not fnmatch.fnmatch(f,'*true*')]
maxb = np.zeros(len(betw_files))
for irun in range(len(betw_files)):
    betws,density2 = myload(betw_files[irun])
    betws = np.array(betws)
    maxb[irun] = betws.max()

maxb.mean(), maxb.std()
# %%
betw_files = [f for f in find(f'betws_dens{dens}_*nu1.5*len0.2*regular*', base_path) if fnmatch.fnmatch(f,'*true*')]
betws,density2 = myload(betw_files[0])
np.array(betws).max()

# %%
adj = get_adj(5-reg_dists, dens,weighted=False)
deg = adj.sum(axis = 0)/(adj.shape[0]-1)
G = nx.from_numpy_matrix(adj)
density2[irun] = nx.density(G) #[i,irun]
betws = list(nx.betweenness_centrality(G).values())#[i,:]
# save and plot betweenness
mysave(base_path,f'betws_dens{dens}_true_'+filter_string[1:-2]+f'.txt',[betws,density2])


# %%
print(np.array(betws).max())

# %%
# expect spurious paths of important betweenness around the equator: poles strongly connected because grid points close, density chosen such that other links barely form
import seaborn as sns

betw_files = find(f'betws_dens{dens}_true_*', base_path)
betws,density2 = myload(betw_files[0])
betws = np.array(betws)
    
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
#betws=myload(base_path+f'realbetws_dens_t2mdaily.txt')
num_points = len(betws)
#var_name = 't2mdaily'

#ar_grid = np.diag(autocorr(data))
thesebetw = np.log10(1+ (num_points-1) * (num_points -2) * betws / 2)
vmin = thesebetw.min() # 0 #np.log10(1+betws[idens,:]).min()
if dens <=0.006:
    vmax = thesebetw.max() #np.log10(1+betws[idens,:]).max() / 2 # 0.025
else:
    vmax = thesebetw.max()
adjust_fontsize(2)
betpl = plot_map_lonlat(lon2,lat2,thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = True)#'RdBu_r'
plt.savefig(base_path+f'betw_map_log_{dens}_gaussgrid_true.pdf')
plt.clf()
adjust_fontsize(3)
betpl = plot_map_lonlat(lon2,lat2,thesebetw,vmin=vmin,vmax=vmax,color=newcols,ctp_projection='EqualEarth',label='Betweenness',earth = True)#'RdBu_r'
plt.savefig(base_path+f'betw_map_log_{dens}_gaussgrid_true_size3.pdf')
plt.clf()


# %%
def get_extr(stat):
    stat_extr = np.zeros_like(stat[:,0,:])
    for i in range(stat.shape[0]):
        for l in range(stat.shape[2]):
            try:
                stat_extr[i,l] = np.where(stat[i,:,l]>0)[0][-1]
            except:
                stat_extr[i,l] = 0
    return stat_extr

betw_extr2 = get_extr(all_betw2)
betw_extr3 = get_extr(all_betw3)

adjust_fontsize(3)
fig, ax = plt.subplots()
for i in range(3):
    ax.plot(binstoplot(betwbins2), all_betw2[i,:,:].mean(axis = 1), label = f'dens={denslist[i]}',linewidth=2.0) # f'dens={np.round(density2[i,:].mean(),4)}'
    ax.fill_between(binstoplot(betwbins2), all_betw2[i,:,:].mean(axis = 1) - 2 *all_betw2[i,:,:].std(axis = 1), all_betw2[i,:,:].mean(axis = 1) + 2 *all_betw2[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(betwbins2), all_betw3[i,:,:].mean(axis = 1),linestyle='--',color = ax.lines[-1].get_color(),linewidth=2.0)
    ax.plot(binstoplot(betwbins2)[int(betw_extr2[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(betwbins2)[int(quantile(betw_extr2[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)
    ax.scatter(binstoplot(betwbins2)[int(quantile(betw_extr3[i,:],alpha = 0.95))],0,marker = 'o',color = ax.lines[-1].get_color(),s=90, facecolors='none') #scatter and , facecolors='none'
ax.legend()
ax.set_xlabel('Betweenness')
ax.set_ylabel('Number of nodes')
ax.set_ylim(-7,100)
#plt.show()
plt.savefig(base_path+f'betweenness3_{similarity}_{filter_string[1:-2]}.pdf')