# %%
from lib2to3.pytree import BasePattern
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
from tueplots import bundles, figsizes
#import seaborn as sn
#from GraphRicciCurvature.OllivierRicci import OllivierRicci
#from GraphRicciCurvature.FormanRicci import FormanRicci

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
# %%
formanbins = np.linspace(-200,100,301)
num_runs = 30
n_estim = 500
n_pergroup = 750
ps = [0.005,0.01,0.05]
frac = 0.1
num_bins = 100

betwbins = np.linspace(0,0.04, num_bins + 1)
all_betw = np.zeros((len(ps), num_bins, num_runs))
all_betw4 = np.zeros((len(ps), num_bins, num_runs))
all_forman = np.zeros((len(ps), len(formanbins)-1, num_runs))
all_forman4 = np.zeros((len(ps), len(formanbins)-1, num_runs))

density = np.zeros((len(ps),num_runs))
base_path = '../../climnet_output/' #'~/Documents/climnet_output/'

def get_extr(stat):
    stat_extr = np.zeros_like(stat[:,0,:])
    for i in range(stat.shape[0]):
        for l in range(stat.shape[2]):
            try:
                stat_extr[i,l] = np.where(stat[i,:,l]>0)[0][-1]
            except:
                stat_extr[i,l] = 0
    return stat_extr

# Stochastic Block Model
# %%
for run in range(num_runs):
    for i in range(len(ps)):
        p = ps[i]
        probs = [[p, frac*p], [frac*p, p]]
        G = nx.stochastic_block_model([n_pergroup,n_pergroup], probs)
        density[i,run] = nx.density(G)
        betwe = nx.betweenness_centrality(G, k = n_estim)
        betwnum = np.histogram(list(betwe.values()), bins = betwbins)[0]
        all_betw[i,:,run] = betwnum
        # rc = FormanRicci(G, verbose="INFO")
        # rc.compute_ricci_curvature()
        # all_forman[i,:,run] = np.histogram(list(nx.get_edge_attributes(rc.G, 'formanCurvature').values()),bins=formanbins)[0]
        G4 = nx.random_degree_sequence_graph([d for num, d in G.degree()],tries=100)
        betwe4 = nx.betweenness_centrality(G4, k = n_estim)
        betwnum4 = np.histogram(list(betwe4.values()), bins = betwbins)[0]
        all_betw4[i,:,run] = betwnum4
        # rc4 = FormanRicci(G4, verbose="INFO")
        # rc4.compute_ricci_curvature()
        # all_forman4[i,:,run] = np.histogram(list(nx.get_edge_attributes(rc4.G, 'formanCurvature').values()),bins=formanbins)[0]
        


all_betw /= 2 * n_pergroup
all_betw4 /= 2 * n_pergroup
# %%
Gadj = nx.adjacency_matrix(G)
plt.hist(Gadj.sum(axis=1),bins=20)
# %%
#rdidcs = np.random.permutation(adj.shape[1])[:500]
plt.spy(nx.adjacency_matrix(G),markersize=0.1)
# %%
plt.spy(nx.adjacency_matrix(G4),markersize=0.1)

# %%
# plt.rcParams["figure.figsize"] = (8*2,6*2)
# plt.rcParams.update({'font.size': 20})
betw_extr = get_extr(all_betw)
betw_extr4 = get_extr(all_betw4)
forman_extr = get_extr(all_forman)
forman_extr4 = get_extr(all_forman4)
# %%
adjust_fontsize(3)
fig,ax = plt.subplots()
for i in range(len(ps)):
    ax.plot(binstoplot(betwbins), all_betw[i,:,:].mean(axis = 1), label = f'dens={np.round(density[i,:].mean(),4)}, p={ps[i]}',linewidth=2.0)
    ax.fill_between(binstoplot(betwbins), all_betw[i,:,:].mean(axis = 1) - 2 *all_betw[i,:,:].std(axis = 1), all_betw[i,:,:].mean(axis = 1) + 2 *all_betw[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(betwbins)[int(betw_extr[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(betwbins)[int(quantile(betw_extr[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)

ax.legend()
ax.set_xlabel('Betweenness')
ax.set_ylabel('Number of nodes')
ax.set_ylim(-0.04,0.2)
plt.savefig(base_path+f'betweenness3_sbm_icml_{num_runs}.pdf')


# %%
fig, ax = plt.subplots()
for i in range(len(ps)):
    ax.plot(binstoplot(betwbins), all_betw4[i,:,:].mean(axis = 1), label = f'dens={np.round(density[i,:].mean(),4)}',linewidth=2.0)
    ax.fill_between(binstoplot(betwbins), all_betw4[i,:,:].mean(axis = 1) - 2 *all_betw4[i,:,:].std(axis = 1), all_betw4[i,:,:].mean(axis = 1) + 2 *all_betw4[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(betwbins)[int(betw_extr4[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(betwbins)[int(quantile(betw_extr4[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)
ax.legend()
ax.set_xlabel('Betweenness')
ax.set_ylabel('Number of nodes')
ax.set_ylim(-0.04,0.2)
#ax.set_ylim(-0.1,1)
#plt.show()
plt.savefig(base_path+f'betweenness3_config.pdf')

# %%
fig,ax = plt.subplots()

for i in range(len(ps)):
    ax.plot(binstoplot(formanbins), all_forman[i,:,:].mean(axis = 1), label = f'dens={np.round(density[i,:].mean(),4)}, p={ps[i]}',linewidth=2.0)
    ax.fill_between(binstoplot(formanbins), all_forman[i,:,:].mean(axis = 1) - 2 *all_forman[i,:,:].std(axis = 1), all_forman[i,:,:].mean(axis = 1) + 2 *all_forman[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(formanbins)[int(forman_extr[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(formanbins)[int(quantile(forman_extr[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)

ax.legend()
ax.set_xlabel('Forman curvature')
ax.set_ylabel('Number of edges')
ax.set_xlim(-120,30)
plt.savefig(base_path+f'forman3_sbm_icml_{num_runs}.pdf')

# %%
fig, ax = plt.subplots()
for i in range(len(ps)):
    ax.plot(binstoplot(formanbins), all_forman4[i,:,:].mean(axis = 1), label = f'dens={np.round(density[i,:].mean(),4)}',linewidth=2.0)
    ax.fill_between(binstoplot(formanbins), all_forman4[i,:,:].mean(axis = 1) - 2 *all_forman4[i,:,:].std(axis = 1), all_forman4[i,:,:].mean(axis = 1) + 2 *all_forman4[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(formanbins)[int(forman_extr4[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(formanbins)[int(quantile(forman_extr4[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)
ax.legend()
ax.set_xlabel('Forman curvature')
ax.set_ylabel('Number of edges')
ax.set_xlim(-120,30)
#plt.show()
plt.savefig(base_path+f'forman3_config.pdf')
# %%
# compute betweenness for Spearman
n_lat = 36 * 2
num_runs = 30
denslist = [0.001,0.01,0.05,0.1]#,0.2]
nu=1.5
len_scale = 0.2
n_time = 100
similarity = 'spearman'
filter_string = f'*matern_nu{nu}_len{len_scale}_{similarity}_ar0_fekete{n_lat}_time{n_time}_*'
betwbins2 = np.linspace(0,0.075, num_bins + 1)

from sklearn.gaussian_process.kernels import Matern
num_points = gridstep_to_numpoints(180/n_lat)
grid = FeketeGrid(num_points = num_points)
lon, lat = grid.grid['lon'], grid.grid['lat']
dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

# %%
find('betws_'+filter_string[1:-2]+'*',base_path)
#find('betws_'+filter_string[1:-2],base_path)
#%%

if find('betws_'+filter_string[1:-2]+'*',base_path) != []:
    all_betw2,all_betw3, density2, density3 = myload(find('betws_'+filter_string[1:-2]+'*',base_path)[0])
else:
    max_runs = len(find(filter_string, base_path+ 'empdata/'))
    if max_runs != 30:
        print('max_runs not 30 but ', max_runs)
    all_betw2 = np.zeros((len(denslist), num_bins, num_runs))
    all_forman2 = np.zeros((len(denslist), len(formanbins)-1,num_runs))
    density2 = np.zeros((len(denslist),num_runs))
    for irun in range(num_runs):
        if max_runs <= irun:
            seed = int(time.time())
            np.random.seed(seed)
            data = diag_var_process(np.zeros(num_points), cov, n_time)
            mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar0_fekete72_time{n_time}_var1_seed{seed}.txt',data)
        else:
            data = myload(find(filter_string, base_path+ 'empdata/')[irun])
        for j in range(data.shape[1]):
            data[:,j] -= data[:,j].mean()
            data[:,j] /= data[:,j].std()
        
        emp_corr = compute_empcorr(data, similarity)
        for i, dens in enumerate(denslist):
            adj = get_adj(emp_corr,dens, weighted = False)
            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            G = nx.from_numpy_matrix(adj)
            density2[i,irun] = nx.density(G)
            betwe = nx.betweenness_centrality(G, k = n_estim)
            betwnum = np.histogram(list(betwe.values()), bins = betwbins2)[0]
            all_betw2[i,:,irun] = betwnum
            #rc = FormanRicci(G, verbose="INFO")
            #rc.compute_ricci_curvature()
            #all_forman2[i,:,irun] = np.histogram(list(nx.get_edge_attributes(rc.G, 'formanCurvature').values()),bins=formanbins)[0]

    all_betw3 = np.zeros((len(denslist), num_bins, 5))
    all_forman3 = np.zeros((len(denslist), len(formanbins)-1))
    density3 = np.zeros((len(denslist),5))
    for irun in range(5):
        if irun == num_runs:
            break
        for i, dens in enumerate(denslist):
            adj = get_adj(cov,dens, weighted = False)
            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            G = nx.from_numpy_matrix(adj)
            #if irun == 0:
            #    rc = FormanRicci(G, verbose="INFO")
            #    rc.compute_ricci_curvature()
            #    all_forman3[i,:] = np.histogram(list(nx.get_edge_attributes(rc.G, 'formanCurvature').values()),bins=formanbins)[0]

            density3[i,irun] = nx.density(G)
            betwe = nx.betweenness_centrality(G, k = n_estim)
            betwnum = np.histogram(list(betwe.values()), bins = betwbins2)[0]
            all_betw3[i,:,irun] = betwnum
    betws = [all_betw2,all_betw3, density2, density3] # add forman?
    mysave(base_path,'betws_'+filter_string[1:-2]+'.txt',betws)
# %%
for i in range(len(denslist)):
    all_forman3[i,:] /= all_forman3[i,:].sum()
    for irun in range(all_forman2.shape[2]):
         all_forman2[i,:,irun] /= all_forman2[i,:,irun].sum()
#%%
find(f'betws_'+filter_string[1:-2]+'*',base_path)

# %%
all_betw2,all_betw3, density2, density3 = myload(find(f'betws_'+filter_string[1:-2]+'*',base_path)[0])
betw_extr2 = get_extr(all_betw2)
betw_extr3 = get_extr(all_betw3)
# forman_extr2 = get_extr(all_forman2)
# forman_extr3 = np.zeros(all_forman3.shape[0])
# for i in range(all_forman3.shape[0]):
#     forman_extr3[i] = np.where(all_forman3[i,:]>0)[0][-1]

# %%
betw_exper = [all_betw,all_betw2,all_betw3,all_betw4,all_forman,all_forman2,all_forman3,all_forman4]
mysave(base_path, 'betweenness_experiment.txt',betw_exper)
betw_extr3.mean(axis=1), betw_extr3.std(axis=1)

# %%
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
plt.savefig(base_path+f'betweenness3_{filter_string[1:-2]}.pdf')

# %%
fig,ax = plt.subplots()

for i in range(len(ps)):
    ax.plot(binstoplot(formanbins), all_forman2[i,:,:].mean(axis = 1), label = f'dens={np.round(density2[i,:].mean(),4)}',linewidth=2.0)
    ax.fill_between(binstoplot(formanbins), all_forman2[i,:,:].mean(axis = 1) - 2 *all_forman2[i,:,:].std(axis = 1), all_forman2[i,:,:].mean(axis = 1) + 2 *all_forman2[i,:,:].std(axis = 1), alpha = 0.4)
    ax.plot(binstoplot(formanbins), all_forman3[i,:],linestyle = '--', linewidth=2.0,color = ax.lines[-1].get_color())
    ax.plot(binstoplot(formanbins)[int(forman_extr2[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
    ax.plot(binstoplot(formanbins)[int(quantile(forman_extr2[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)
    ax.scatter(binstoplot(formanbins)[int(forman_extr3[i])],0,marker = 'o',color = ax.lines[-1].get_color(),s=50, facecolors='none') #scatter and , facecolors='none'

ax.legend(loc='upper left')
ax.set_xlabel('Forman curvature')
ax.set_ylabel('PDF')
plt.savefig(base_path+f'forman3_{similarity}_{filter_string[1:-1]}.pdf')


# %%
#plt.rcParams.update(figsizes.icml2022_full(ncols = 2))
#fig, ax = plt.subplots(nrows=1, ncols=2)

# for i in range(len(ps)):
#     ax[0].plot(binstoplot(betwbins), all_betw[i,:,:].mean(axis = 1), label = f'dens={np.round(density[i,:].mean(),4)}, p={ps[i]}',linewidth=2.0)
#     ax[0].fill_between(binstoplot(betwbins), all_betw[i,:,:].mean(axis = 1) - 2 *all_betw[i,:,:].std(axis = 1), all_betw[i,:,:].mean(axis = 1) + 2 *all_betw[i,:,:].std(axis = 1), alpha = 0.4)
#     ax[0].plot(binstoplot(betwbins)[np.where(all_betw[i,:,:].mean(axis=1)>0)[0][-5:]],all_betw[i,np.where(all_betw[i,:,:].mean(axis=1)>0)[0][-5:],:].mean(axis=1),'x',color = ax.lines[-1].get_color(),markersize=12)

# ax[0].legend()
# ax[0].set_xlabel('Betweenness')
# ax[0].set_ylabel('# points')

# for i in range(len(denslist)):
#     ax[1].plot(binstoplot(betwbins2), all_betw2[i,:,:].mean(axis = 1), label = f'dens={np.round(density2[i,:].mean(),4)}',linewidth=2.0)
#     ax[1].fill_between(binstoplot(betwbins2), all_betw2[i,:,:].mean(axis = 1) - 2 *all_betw2[i,:,:].std(axis = 1), all_betw[i,:,:].mean(axis = 1) + 2 *all_betw[i,:,:].std(axis = 1), alpha = 0.4)
#     ax[1].plot(binstoplot(betwbins2)[np.where(all_betw2[i,:,:].mean(axis=1)>0)[0][-5:]],all_betw[i,np.where(all_betw[i,:,:].mean(axis=1)>0)[0][-5:],:].mean(axis=1),'x',color = ax.lines[-1].get_color(),markersize=12)

# ax[1].legend()
# ax[1].set_xlabel('Betweenness')
# ax[1].set_ylabel('# points')
# plt.show()
# #plt.savefig(base_path+f'betweenness_{similarity}_{filter_string}.pdf', dpi = 300)
# %%
G4 = nx.configuration_model([d for num, d in G.degree()])
G4
# %%
i = 0
fig,ax = plt.subplots()
ax.plot(binstoplot(formanbins), all_forman2[i,:,:].mean(axis = 1), label = f'dens={np.round(density2[i,:].mean(),4)}',linewidth=2.0)
ax.fill_between(binstoplot(formanbins), all_forman2[i,:,:].mean(axis = 1) - 2 *all_forman2[i,:,:].std(axis = 1), all_forman2[i,:,:].mean(axis = 1) + 2 *all_forman2[i,:,:].std(axis = 1), alpha = 0.4)
ax.plot(binstoplot(formanbins), all_forman3[i,:],linestyle = '--', linewidth=2.0,color = ax.lines[-1].get_color())
ax.plot(binstoplot(formanbins)[int(forman_extr2[i,:].mean())],0,'x',color = ax.lines[-1].get_color(),markersize=12)
ax.plot(binstoplot(formanbins)[int(quantile(forman_extr2[i,:],alpha = 0.95))],0,marker = 4,color = ax.lines[-1].get_color(),markersize=12)
ax.scatter(binstoplot(formanbins)[int(forman_extr3[i])],0,marker = 'o',color = ax.lines[-1].get_color(),s=50, facecolors='none') #scatter and , facecolors='none'

ax.legend(loc='upper left')
ax.set_xlabel('Forman curvature')
ax.set_ylabel('PDF')

# %%
adj = get_adj(emp_corr,0.05, weighted = False)
plt.hist(dists[adj!=0], bins = 30)

# %%
rd_idcs = np.random.permutation(len(adj != 0))[:10000]

# plt.scatter(dists[adj!=0][rd_idcs], cov[adj!=0][rd_idcs])
plt.scatter(dists[rd_idcs], cov[rd_idcs])
# %%
kernel = 1.0 * Matern(length_scale=0.1, nu=1.5)
xs = np.linspace(0,180,100)
ys = kernel(spherical2cartesian(xs,np.zeros_like(xs)))[0,:]
plt.plot(xs, ys)
# %%
