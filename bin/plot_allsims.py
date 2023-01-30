# %%
#import networkx as nx
import os
import numpy as np
#import scipy.interpolate as interp
#from scipy import stats, ndimage
#import scipy
import tueplots
#import xarray as xr
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import revised_mi
import time
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
curr_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'

n_lat = 18 * 2
typ ='threshold'
corr_method='spearman'
weighted = False # iterate over

ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True

ar = 0
ar2 = None
var = 10

robust_tolerance = None # need None for true graphstats
n_perm = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600]#,1200]


#filter_string = f'nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'
# %%
#grid_helper and calc_true
nu=1.5
len_scale = 0.1
exec(open("grid_helper.py").read())
# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points
#exec(open("calc_true.py").read())
#old: true_dens, true_degs, true_lls, true_ccs, true_ccws, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_ccws, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
#tele: true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2, true_madw, true_shufflemadw = true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2, true_madw, true_shufflemadw
#true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2 = true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99

if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')


# %%
import matplotlib.pyplot as plt
density = 0.005
nu = 0.5 
len_scale = 0.2
#n_lat = 2*18
n_time = 100
ar = 0
ar2 = None

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

print('Computing covariance matrix.')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
    cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
else:
    cov = kernel(spherical2cartesian(lon,lat))
    mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
ar_coeff = np.zeros(len(cov))

thisnam = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')][0]
seed = int(thisnam.split('_seed',1)[1].split('.txt',1)[0])
seed
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

# %%
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


max_dist = 5-get_adj(5-dists,density)[get_adj(5-dists,density)!=0].min()
vmin = 0
emp_corr = compute_empcorr(data, 'pearson')
adj = get_adj(emp_corr, density, weighted=False)
vmax = adj.sum(axis=1).max()

# %%
pemp_corr = compute_empcorr(data,similarity = 'pearson')
padj = get_adj(pemp_corr, density, weighted=False)
lwemp_corr = compute_empcorr(data, similarity='LWlin')
lwadj = get_adj(lwemp_corr, density, weighted=False)

# for d=1483 we find LW better, but for d=5981 not?! (for large lenscale)
np.linalg.norm(cov - pemp_corr), np.linalg.norm(cov - pemp_corr, ord = 2),np.linalg.norm(cov - lwemp_corr), np.linalg.norm(cov - lwemp_corr, ord = 2)

# %%
# plot 0.5%-graphs of all estimators
for corr_method in ['pearson', 'spearman', 'binMI', 'HSIC', 'LWlin','LW','BI-KSG']: #, 'ES'
    # if find(f'linkplot_empcorr_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*.txt', base_path) != []:
    #     emp_corr = myload(find(f'linkplot_empcorr_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*.txt', base_path)[0])
    #     adj = get_adj(emp_corr, density, weighted=False)
    #     num_falselinks = (dists[adj!=0] > max_dist).sum()
    #     thisname = find(f'linkplot_empcorr_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*.txt', base_path)[0].split('linkplot_empcorr_',1)[1]
    # else:
    # if corr_method == 'BI-KSG':
    #     emp_corr = myload(thisnam)# wrong shape
    #     # find nearest neighbors for each grid point of the coarse grid or compute empcorr
    #     TODO
    # else:
    emp_corr = compute_empcorr(data, similarity=corr_method)
    adj = get_adj(emp_corr, density, weighted=False)
    num_falselinks = (dists[adj!=0] > max_dist).sum()
    thisname = f'{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_fl{num_falselinks}_{seed}.pdf'
    mysave(base_path, f'linkplot_empcorr_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{seed}.txt',emp_corr)
    
    adjust_fontsize(2)
    edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    edge_plot['ax'].set_rasterization_zorder(0)
    plt.savefig(base_path+f'linkplots/rasterlinks_'+ thisname[:-4]+'.pdf',rasterize = True)
    edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    plt.savefig(base_path+f'linkplots/rasterlinkspng_'+ thisname[:-4]+'.png')
    adjust_fontsize(3)
    edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    edge_plot['ax'].set_rasterization_zorder(0)
    plt.savefig(base_path+f'linkplots/rasterlinks3_'+ thisname[:-4]+'.pdf',rasterize = True)
    edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
    plt.savefig(base_path+f'linkplots/rasterlinkspng3_'+ thisname[:-4]+'.png')



# %%
biksg = myload(thisnam)
biksg.shape
# %%
data = diag_var_process(ar_coeff, cov, 1000)
data = np.sqrt(var) * data
exp_data = np.exp(np.sqrt(var) * data)
data2 = exp_data
# save with nam in name
for j in range(data.shape[1]):
    data[:,j] -= data[:,j].mean()
    data[:,j] /= data[:,j].std()
    data2[:,j] -= exp_data[:,j].mean()
    data2[:,j] /= exp_data[:,j].std()
emp_corr = compute_empcorr(data, similarity='ES')
adj = get_adj(emp_corr, density, weighted=False)
num_falselinks = (dists[adj!=0] > max_dist).sum()
thisname = f'ES_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_var1_fl{num_falselinks}_{seed}.pdf'
mysave(base_path, f'linkplot_empcorr_ES_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_{seed}.txt',emp_corr)
adjust_fontsize(2)
edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
edge_plot['ax'].set_rasterization_zorder(0)
plt.savefig(base_path+f'linkplots/links_'+ thisname,rasterize=True)
adjust_fontsize(3)
edge_plot = plot_deg_with_edges(grid.grid, adj, vmin = vmin, vmax=vmax, transform = ctp.crs.Geodetic(), label = 'Degree')
edge_plot['ax'].set_rasterization_zorder(0)
plt.savefig(base_path+f'linkplots/rasterlinks3_'+ thisname,rasterize=True)
# %%
deg = adj.sum(axis = 0)/(adj.shape[0]-1)
deg.mean()
# %%
emp_corr = myload(find(f'linkplot_empcorr_BIKSG_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*.txt', base_path)[0])
emp_corr3 = myload(find(f'linkplot_empcorr_LWlin_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*.txt', base_path)[0])
if np.all(emp_corr3 == emp_corr):
    raise ValueError('Empcorr calculation still fails!')

# %%

for plinkname in find(f'plink_*', base_path+'plink/'):
    plink = myload(plinkname)
    savename = plinkname.split('plink_',1)[1][:-4]
    nu = np.float64(plinkname.split('nu',1)[1].split('_',1)[0])
    len_scale = plinkname.split('len',1)[1].split('_',1)[0]
    corr_method = plinkname.split('plink_',1)[1].split('_',1)[0]
    n_lat = np.int64(plinkname.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    num_lines = all_degs.shape[0]
    fig,ax = plt.subplots()
    for lineidx in range(num_lines):
        ax.plot(binstoplot(truecorrbins), plink[lineidx,:], label = f'dens={denslist[lineidx]}')
    ax.legend()
    ax.set_ylabel('Link probability')
    ax.set_xlabel('True correlation')
    ax.set_xlim(0,1)
    #plt.savefig(base_path+'plink/plot_linkprob_'+savename+'.pdf')

# %%
import fnmatch
# compose fdr
gslist = []
for filename in find('allfdr_*', base_path+ 'fdr/',nosub=True):
    hyperparams = filename.split('fdr_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)


# %%
n_lat = 72
denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]

for nu in nus:
    for len_scale in len_scales:
        filter_string = f'allfdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'
        if find(filter_string, base_path+'fdr/') == []:
            print(f'No FDR for {nu} and {len_scale}')
            continue
        filenames = []
        iteration = 0
        while len(filenames) < len(find(filter_string, base_path+'fdr/')):
            iteration += 1
            for fdrname in find(filter_string, base_path+'fdr/'):
                corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
                if corr_method == 'pearson' and len(filenames) == 0:
                    filenames.append(fdrname)
                elif corr_method == 'spearman' and len(filenames) == 1:
                    filenames.append(fdrname)
                elif corr_method == 'LWlin' and len(filenames) == 2:
                    filenames.append(fdrname)
                elif corr_method == 'binMI' and len(filenames) == 3:
                    filenames.append(fdrname)
                elif corr_method == 'BI-KSG' and len(filenames) == 4:
                    filenames.append(fdrname)
                elif corr_method == 'HSIC' and len(filenames) == 5:
                    filenames.append(fdrname)
                elif corr_method == 'ES' and len(filenames) == 6:
                    continue
                #elif corr_method == 'ES' and len(filenames) == 6:
                #    filenames.append(fdrname)
            if iteration > 1000:
                break
        fig,ax = plt.subplots()
        for fdrname in filenames:
            thisfdr = myload(fdrname)
            if thisfdr.shape[0] != len(denslist):
                print(fdrname + f' has length {thisfdr.shape[0]}')
                continue
            corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
            savename = fdrname.split(f'{corr_method}_',1)[1][:-4]
            print(corr_method,thisfdr.mean(axis=1))
            if corr_method == 'pearson':
                col = 'tab:blue'
                corr_method = 'Pearson/LW'
            elif corr_method == 'spearman':
                col = 'tab:orange'
                corr_method = 'Spearman'
            elif corr_method == 'LWlin':
                continue
                col = 'tab:green'
                corr_method = 'LW'
            elif corr_method == 'binMI':
                col = 'tab:green'
                corr_method = 'Binned MI'
            elif corr_method == 'BI-KSG':
                col = 'tab:red'
                corr_method = 'BI-KSG MI'
            elif corr_method == 'HSIC':
                col = 'tab:purple'
            elif corr_method == 'ES':
                col = 'tab:brown'#'tab:pink'
            ax.plot(denslist, thisfdr.mean(axis=1), label = f'{corr_method}', color = col)
            ax.fill_between(denslist, [quantile(thisfdr[idens,:],alpha=0.025) for idens in range(len(denslist))],[quantile(thisfdr[idens,:],alpha=0.975) for idens in range(len(denslist))],color=col, alpha = 0.4)
        ax.legend()
        ax.set_ylabel('FDR')
        ax.set_xlabel('Density')
        #ax.set_ylim(0,0.2)
        xlim = 0.05
        ax.set_xlim(0,xlim)
        if find(filter_string, base_path+'fdr/') != []:
            plt.savefig(base_path+f'fdr/plot_fdr_new_noES_{len(denslist)}_xlim{xlim}_'+savename+'.pdf')

# %%
# joint plot for paper

fig,axs = plt.subplots(1,3,figsize=(3*onefigsize[0],1.2*onefigsize[1]))

n_lat = 72
denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]

nu = 0.5
len_scale = 0.1
filter_string = f'allfdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'
if find(filter_string, base_path+'fdr/') == []:
    raise ValueError(f'No FDR for {nu} and {len_scale}')
filenames = []
iteration = 0
while len(filenames) < len(find(filter_string, base_path+'fdr/')):
    iteration += 1
    for fdrname in find(filter_string, base_path+'fdr/'):
        corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
        if corr_method == 'pearson' and len(filenames) == 0:
            filenames.append(fdrname)
        elif corr_method == 'spearman' and len(filenames) == 1:
            filenames.append(fdrname)
        elif corr_method == 'LWlin' and len(filenames) == 2:
            filenames.append(fdrname)
        elif corr_method == 'binMI' and len(filenames) == 3:
            filenames.append(fdrname)
        elif corr_method == 'BI-KSG' and len(filenames) == 4:
            filenames.append(fdrname)
        elif corr_method == 'HSIC' and len(filenames) == 5:
            filenames.append(fdrname)
        elif corr_method == 'ES' and len(filenames) == 6:
            continue
        #elif corr_method == 'ES' and len(filenames) == 6:
        #    filenames.append(fdrname)
    if iteration > 1000:
        break

for fdrname in filenames:
    thisfdr = myload(fdrname)
    if thisfdr.shape[0] != len(denslist):
        print(fdrname + f' has length {thisfdr.shape[0]}')
        continue
    corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
    savename = fdrname.split(f'{corr_method}_',1)[1][:-4]
    print(corr_method,thisfdr.mean(axis=1))
    if corr_method == 'pearson':
        col = 'tab:blue'
        corr_method = 'Pearson/LW'
    elif corr_method == 'spearman':
        col = 'tab:orange'
        corr_method = 'Spearman'
    elif corr_method == 'LWlin':
        continue
        col = 'tab:green'
        corr_method = 'LW'
    elif corr_method == 'binMI':
        col = 'tab:green'
        corr_method = 'Binned MI'
    elif corr_method == 'BI-KSG':
        col = 'tab:red'
        corr_method = 'BI-KSG MI'
    elif corr_method == 'HSIC':
        col = 'tab:purple'
    elif corr_method == 'ES':
        col = 'tab:brown'#'tab:pink'
    axs[0].plot(denslist, thisfdr.mean(axis=1), label = f'{corr_method}', color = col)
    axs[0].fill_between(denslist, [quantile(thisfdr[idens,:],alpha=0.025) for idens in range(len(denslist))],[quantile(thisfdr[idens,:],alpha=0.975) for idens in range(len(denslist))],color=col, alpha = 0.4)
axs[0].legend()
axs[0].set_ylabel('FDR')
axs[0].set_xlabel('Density')
#axs[0].set_ylim(0,0.2)
xlim = 0.05
axs[0].set_xlim(0,xlim)

nu = 1.5
len_scale = 0.2
filter_string = f'allfdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'
if find(filter_string, base_path+'fdr/') == []:
    raise ValueError(f'No FDR for {nu} and {len_scale}')
filenames = []
iteration = 0
while len(filenames) < len(find(filter_string, base_path+'fdr/')):
    iteration += 1
    for fdrname in find(filter_string, base_path+'fdr/'):
        corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
        if corr_method == 'pearson' and len(filenames) == 0:
            filenames.append(fdrname)
        elif corr_method == 'spearman' and len(filenames) == 1:
            filenames.append(fdrname)
        elif corr_method == 'LWlin' and len(filenames) == 2:
            filenames.append(fdrname)
        elif corr_method == 'binMI' and len(filenames) == 3:
            filenames.append(fdrname)
        elif corr_method == 'BI-KSG' and len(filenames) == 4:
            filenames.append(fdrname)
        elif corr_method == 'HSIC' and len(filenames) == 5:
            filenames.append(fdrname)
        elif corr_method == 'ES' and len(filenames) == 6:
            continue
        #elif corr_method == 'ES' and len(filenames) == 6:
        #    filenames.append(fdrname)
    if iteration > 1000:
        break

for fdrname in filenames:
    thisfdr = myload(fdrname)
    if thisfdr.shape[0] != len(denslist):
        print(fdrname + f' has length {thisfdr.shape[0]}')
        continue
    corr_method = fdrname.split('_nu',1)[0].split('_',10)[-1]
    savename = fdrname.split(f'{corr_method}_',1)[1][:-4]
    print(corr_method,thisfdr.mean(axis=1))
    if corr_method == 'pearson':
        col = 'tab:blue'
        corr_method = 'Pearson/LW'
    elif corr_method == 'spearman':
        col = 'tab:orange'
        corr_method = 'Spearman'
    elif corr_method == 'LWlin':
        continue
        col = 'tab:green'
        corr_method = 'LW'
    elif corr_method == 'binMI':
        col = 'tab:green'
        corr_method = 'Binned MI'
    elif corr_method == 'BI-KSG':
        col = 'tab:red'
        corr_method = 'BI-KSG MI'
    elif corr_method == 'HSIC':
        col = 'tab:purple'
    elif corr_method == 'ES':
        col = 'tab:brown'#'tab:pink'
    axs[1].plot(denslist, thisfdr.mean(axis=1), label = f'{corr_method}', color = col)
    axs[1].fill_between(denslist, [quantile(thisfdr[idens,:],alpha=0.025) for idens in range(len(denslist))],[quantile(thisfdr[idens,:],alpha=0.975) for idens in range(len(denslist))],color=col, alpha = 0.4)
axs[1].legend()
axs[1].set_ylabel('FDR')
axs[1].set_xlabel('Density')
#axs[0].set_ylim(0,0.2)
xlim = 0.05
axs[1].set_xlim(0,xlim)

corr_method= 'LWlin'
thesenus, theselen_scales,idcs = [], [],[]
frobs1,frobs2,minfrobs1,maxfrobs1,minfrobs2,maxfrobs2 = [[] for _ in range(6)]
l2s1,l2s2,minl2s1,maxl2s1,minl2s2,maxl2s2 = [[] for _ in range(6)]
thesenames = find('allempmatrixdist*',base_path)
if corr_method == 'LWlin':
    corridx = 0
elif corr_method == 'LW':
    corridx = 1
# for distname in find('allempmatrixdist*',base_path):
#     if corr_method == distname.split('allempmatrixdist_',1)[1].split('_',10)[0]:
#         thesenames.append(distname)
for idx, distname in enumerate(thesenames):
    frob,l2 = myload(distname)
    savename = distname.split('allempmatrixdist_',1)[1][:-4]
    nu = np.float64(distname.split('nu',1)[1].split('_',1)[0])
    len_scale = distname.split('len',1)[1].split('_',1)[0]
    if len_scale == '0.05':
        idx1 = 0
    elif len_scale == '0.1':
        idx1 = 1
    else:
        idx1 = 2
    if nu == 0.5:
        idx2 = 0
    else:
        idx2 = 1
    idcs.append(2*idx1+idx2)
    thesenus.append(nu)
    theselen_scales.append(len_scale)
    # make single ax.errorbar with lw=0 for legend
    frobs1.append(frob[-1,:].mean())
    frobs2.append(frob[corridx,:].mean())
    minfrobs1.append(frob[-1,:].min())
    maxfrobs1.append(frob[-1,:].max())
    minfrobs2.append(frob[corridx,:].min())
    maxfrobs2.append(frob[corridx,:].max())
    l2s1.append(l2[-1,:].mean())
    l2s2.append(l2[corridx,:].mean())
    minl2s1.append(l2[-1,:].min())
    maxl2s1.append(l2[-1,:].max())
    minl2s2.append(l2[corridx,:].min())
    maxl2s2.append(l2[corridx,:].max())
    #ax.errorbar(idx,frob[-1,:].mean(),np.array([frob[-1,:].min(),frob[-1,:].max()]).reshape((2,1)), fmt='.k', color = 'tab:blue', label = 'emp. pearson')
    #ax.errorbar(idx, frob[0,:].mean(),np.array([frob[0,:].min(),frob[0,:].max()]).reshape((2,1)), fmt='.k', color = 'tab:orange', label = corr_method)
idcs = np.array(idcs)
#axs[2].boxplot([frob[-1,:],frob[corridx,:]]), each column one list entry containing data
axs[2].errorbar(idcs-0.13,frobs1,np.array([minfrobs1,maxfrobs1]), fmt='.', capsize=4, color = 'tab:blue', label = 'Emp. Pearson')
axs[2].errorbar(idcs+0.13,frobs2,np.array([minfrobs2,maxfrobs2]), fmt='.', capsize=4, color = 'tab:orange', label = 'LW')
axs[2].set_xticks(idcs)
axs[2].set_xticklabels([f'nu={thesenus[idx]},\n l={theselen_scales[idx]}' for idx in range(len(thesenames))])
axs[2].set_ylabel('Frobenius error')
axs[2].legend()

axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path + f'joint_fdr_frob_plot.pdf')

# %%
filter_string = f'allplink*'
find(filter_string, base_path+'plink/')
# %%
filter_string = f'allplink_*'
for plinkname in find(filter_string, base_path+'plink/'):
    thisplink = myload(plinkname)
    corr_method = plinkname.split('_nu',1)[0].split('_',10)[-1]
    savename = plinkname.split('allplink_',1)[1][:-4]
    print(savename)
    fig,ax = plt.subplots()
    for lineidx in range(thisplink.shape[0]):
        ax.plot(binstoplot(truecorrbins), thisplink[lineidx,:], label = f'dens={denslist[lineidx]}')
    ax.legend()
    ax.set_ylabel('Link probability')
    ax.set_xlabel('True correlation')
    ax.set_xlim(0,1)
    if find(filter_string, base_path+'plink/') != []:
        plt.savefig(base_path+'plink/plot_plink_'+savename+'.pdf')
# %%
# n_lat = 72
# num_points = gridstep_to_numpoints(180/n_lat)
# for nu in nus:
#     for len_scale in len_scales:
#         filter_string = f'fdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'
#         fig,ax = plt.subplots()
#         for fdrname in find(filter_string, base_path+'fdr/'):
#             fdr = myload(fdrname)
#             corr_method = fdrname.split('fdr_',1)[1].split('_',1)[0]
#             savename = fdrname.split(f'{corr_method}_',1)[1][:-4]
#             print(corr_method,fdr.mean(axis=1))
#             ax.plot(denslist, fdr.mean(axis=1), label = f'{corr_method}')
#             ax.fill_between(denslist, [quantile(fdr[idens,:],alpha=0.025) for idens in range(len(denslist))],[quantile(fdr[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
#         ax.legend()
#         ax.set_ylabel('FDR')
#         ax.set_xlabel('Density')
#         ax.set_ylim(0,0.2)
#         #ax.set_xlim(0,1)
#         if find(filter_string, base_path+'fdr/') != []:
            #plt.savefig(base_path+'fdr/plot_fdr_'+savename+'.pdf')


# %%
# n_lat = 72
# nu = 0.5
# len_scale = 0.1
# ar = 0
# num_points = gridstep_to_numpoints(180/n_lat)
# fig,ax = plt.subplots()
# filter_string = f'allfdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'

# for fdrname in find(filter_string, base_path+'fdr/'):
#     allfdr = myload(fdrname)
#     for idx in range(allfdr.shape[1]):
#         if np.all(allfdr[:,idx] == np.zeros_like(allfdr[:,idx])):
#             allfdr = np.delete(allfdr,idx,1)
#     corr_method = fdrname.split(f'_nu{nu}',1)[0].split('_',10)[-1]
#     savename = fdrname.split(f'{corr_method}_',1)[1][:-4]
#     print(corr_method,allfdr.mean(axis=1))
#     ax.plot(denslist, allfdr.mean(axis=1), label = f'{corr_method}')
#     ax.fill_between(denslist, [quantile(allfdr[idens,:],alpha=0.025) for idens in range(len(denslist))],[quantile(allfdr[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
# ax.legend()
# ax.set_ylabel('FDR')
# ax.set_xlabel('Density')
#ax.set_ylim(0,0.2)
#ax.set_xlim(0,1)
#if find(filter_string, base_path+'fdr/') != []:
#    plt.savefig(base_path+'fdr/plot_fdr_'+savename+'.pdf')

# %%




#plt.hist(dists[np.where(adj3!= adj)[0],np.where(adj3!= adj)[1]], bins = 30)
# %%
filter_string = f'allfdr*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_*'

fdrname = find(filter_string, base_path+'fdr/')[0]
fdrname.split(f'_nu{nu}',1)[0].split('_',10)[-1]

# %%
for idx in range(allfdr.shape[1]):
    print(idx)

# %%
grid_step_lon=5
num_points = gridstep_to_numpoints(grid_step_lon)
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
finenum_points = gridstep_to_numpoints(grid_step_lon/2)
finegrid = FeketeGrid(num_points = finenum_points)
flon, flat = finegrid.grid['lon'], finegrid.grid['lat']

imin = []
for ipt in range(len(lat)):
    imin.append(np.argmin([gdistance((lon[ipt],lat[ipt]),(flon[i2],flat[i2])) for i2 in range(len(flon))]))
mysave(base_path,f'nearest_gridpoints_fekete1483_fekete5981.txt',imin)

# %%
plt.hist([gdistance((lon[ipt],lat[ipt]),(flon[imin[ipt]],flat[imin[ipt]])) for ipt in range(len(lon))],40)
