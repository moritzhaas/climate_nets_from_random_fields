
# %%
import networkx as nx
import os
import numpy as np
import scipy.interpolate as interp
from scipy import stats, ndimage
import scipy
import tueplots
import xarray as xr
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import revised_mi
import time
from sklearn.gaussian_process.kernels import Matern

start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
adjust_fontsize(3)
curr_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'

n_lat = 18 * 4
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
n_time = 500
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5 # irrelevant
len_scale = 0.1 # irrelevant

denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]


filter_string = f'nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'
# %%
#grid_helper and calc_true
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
    plt.tight_layout()
    return axs
#  %%
find(f'allgraphstats_*', base_path+'graphstats/')
# %%
for hyperparams in find(f'allgraphstats_*', base_path+'graphstats/'):
    print(filter_string in hyperparams)
# %%
degbinsw = np.linspace(0,0.2, num_bins+1)
degbinsfine = np.linspace(0,0.3, num_bins+1)
for graphstatname in find(f'allgraphstats_*time{n_time}*', base_path+'graphstats/'):
    # all_dens, all_degs, all_lls, all_ccs,all_ccws, all_spls, plink,all_llquant1,all_llquant2=myload(graphstatname)
    all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = myload(graphstatname)
    savename = graphstatname.split('graphstats_',1)[1][:-4]#+'size3'
    adjust_fontsize(2)
    nu = np.float64(graphstatname.split('nu',1)[1].split('_',1)[0])
    len_scale = np.float64(graphstatname.split('len',1)[1].split('_',1)[0])
    corr_method = graphstatname.split('igrf_',1)[1].split('_',1)[0]
    if (corr_method != 'pearson') and (corr_method != 'BI-KSG'):
        print('Uses degbins and not degbinssparse, degbinsw.')
        continue
    typ = graphstatname.split(corr_method+'_',1)[1].split('_',1)[0]
    # weighted = np.bool_(graphstatname.split(typ+'_w',1)[1].split('_',1)[0])
    n_lat = np.int64(graphstatname.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
    if corr_method == 'BI-KSG':
        ranks = True
    else:
        ranks = False
    #exec(open("calc_true.py").read())
    #true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
    try:
        true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2 = myload(base_path+f'truestats/truegraphstats_matern{nu}_{len_scale}_{typ}_{len(ks)}_ranks{ranks}.txt')
    except:
        print(graphstatname, ' no true stats available.')
        continue
    num_lines = all_degs.shape[0]
    thesedens = np.round(all_dens.mean(axis=1),3)
    all_degs *= num_points
    all_degws *= num_points
    all_ccs *= num_points
    all_ccws *= num_points
    true_degs *= num_points
    true_degws *= num_points
    true_ccs *= num_points
    true_ccws *= num_points
    if corr_method == 'pearson':
        for id, density in enumerate([0.001,0.01,0.05,0.1]):
            if typ == 'knn':
                k = ks[id]
                adj = knn_adj(np.abs(cov),k,weighted=False)
            else:
                adj = get_adj(cov,density,weighted=False) 
            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            degnum = np.histogram(deg, bins=degbinsfine)[0]
            true_degs[id,:] = degnum
    
    fig,ax = plt.subplots(2,3, figsize=(3*onefigsize[0],2*onefigsize[1]))
    for lineidx in range(1,num_lines):    
        ax[0,0].plot(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,0].fill_between(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1) - 2 *all_degs[lineidx,:,:].std(axis = 1), all_degs[lineidx,:,:].mean(axis = 1) + 2 *all_degs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,0].plot(degbinssparse[np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[0,0].lines[-1].get_color())
        ax[0,0].plot(degbinssparse[int(np.maximum(0,np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[0,0].lines[-1].get_color())
        ax[0,0].plot(degbinsfine[:-1],true_degs[lineidx,:], linestyle = '--', color = ax[0,0].lines[-1].get_color())
    ax[0,0].legend()
    ax[0,0].set_ylabel(f'Number of nodes')
    ax[0,0].set_xlabel(f'Normalized degree')
    ax[0,0].set_xlim(0,0.25)
    if not os.path.exists(base_path+'graphstatplots/'):
        os.mkdir(base_path+'graphstatplots/')
    ##plt.savefig(base_path+'graphstatplots/plot_deg_'+savename+'.pdf')

    if corr_method == 'pearson':
        for id, density in enumerate([0.001,0.01,0.05,0.1]):
            if typ == 'threshold':
                adj = get_adj(cov, density, weighted=True)
            else:
                k = ks[id]
                adj = knn_adj(np.abs(cov),k,weighted=True)
            if ranks: # use for MI..
                adj = rank_matrix(adj)
                adj /= (adj.max()+1)
            adj = get_adj(cov,density,weighted=True) 
            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            degnum = np.histogram(deg, bins=degbinsw)[0]
            true_degws[id,:] = degnum
    #fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax[1,0].plot(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,0].fill_between(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1) - 2 *all_degws[lineidx,:,:].std(axis = 1), all_degws[lineidx,:,:].mean(axis = 1) + 2 *all_degws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,0].plot(degbinsw[np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,0].lines[-1].get_color())
        ax[1,0].plot(degbinsw[int(np.maximum(0,np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[1,0].lines[-1].get_color())
        ax[1,0].plot(degbinsw[:-1],true_degws[lineidx,:], linestyle = '--', color = ax[1,0].lines[-1].get_color())
    ax[1,0].legend()
    ax[1,0].set_ylabel(f'Number of nodes')
    ax[1,0].set_xlabel(f'Normalized weighted degree')
    ax[1,0].set_xlim(0,0.1)
    #plt.savefig(base_path+'graphstatplots/plot_degw_'+savename+'.pdf')

    #fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):
        ax[0,2].plot(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,2].fill_between(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1) - 2 *all_lls[lineidx,:,:].std(axis = 1), all_lls[lineidx,:,:].mean(axis = 1) + 2 *all_lls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,2].plot(all_llquant1[lineidx,:].mean(),0,'x',color = ax[0,2].lines[-1].get_color())
        ax[0,2].plot(all_llquant2[lineidx,:].mean(),0,'o',color = ax[0,2].lines[-1].get_color())
        upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
        ax[0,2].plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax[0,2].lines[-1].get_color())
    ax[0,2].legend()
    ax[0,2].set_ylabel(f'Number of links')
    ax[0,2].set_xlabel(f'Distance (in radians)')
    ymax = all_lls.max()
    ax[0,2].set_ylim(-0.1*ymax,1.1*ymax)
    ax[0,2].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax[0,2].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    #plt.savefig(base_path+'graphstatplots/plot_ll_'+savename+'.pdf')


    #fig,ax = plt.subplots()
    #ax.plot(denslist[:all_lls.shape[0]], fdr(all_lls,true_lls[:all_lls.shape[0]]).mean(axis=1))
    #ax.fill_between(denslist[:all_lls.shape[0]], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.025) for idens in range(all_lls.shape[0])], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.975) for idens in range(all_lls.shape[0])], alpha = 0.4)
    # ax.set_ylabel(f'False discovery rate')
    # ax.set_xlabel(f'Density')
    # ax.set_ylim(0,1)
    # plt.savefig(base_path+'graphstatplots/plot_fdr_'+savename+'.pdf')

    #fig,ax = plt.subplots()
    # for lineidx in range(1,num_lines):
    #     ax.plot(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
    #     ax.fill_between(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1) - 2 *all_llws[lineidx,:,:].std(axis = 1), all_llws[lineidx,:,:].mean(axis = 1) + 2 *all_llws[lineidx,:,:].std(axis = 1), alpha = 0.4)
    #     ax.plot(all_llwquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
    #     ax.plot(all_llwquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
    #     upper_ll = llbins[np.where(true_llws[lineidx,:]>0)[0][-1]]
    #     ax.plot((upper_ll,upper_ll), (0,100000), linestyle = '--', color = ax.lines[-1].get_color())
    # ax.legend()
    # ax.set_ylabel(f'Edge weight')
    # ax.set_xlabel(f'Distance (in radians)')
    # ymax = all_llws.max()
    # ax.set_ylim(-0.1*ymax,1.1*ymax)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    # #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    # plt.savefig(base_path+'graphstatplots/plot_llw_'+savename+'.pdf')

    
    #fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax[0,1].plot(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,1].fill_between(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1) - 2 *all_ccs[lineidx,:,:].std(axis = 1), all_ccs[lineidx,:,:].mean(axis = 1) + 2 *all_ccs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,1].plot(degbins[np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[0,1].lines[-1].get_color())
        ax[0,1].plot(degbins[int(np.maximum(0,np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[0,1].lines[-1].get_color())
        ax[0,1].plot(degbins[:-1],true_ccs[lineidx,:], linestyle = '--', color = ax[0,1].lines[-1].get_color())
    ax[0,1].legend()
    ax[0,1].set_ylabel(f'Number of nodes')
    ax[0,1].set_xlabel(f'Clustering coefficient')
    ymax = all_ccs.max()
    ax[0,1].set_ylim(-0.1*ymax,1.1*ymax)
    #plt.savefig(base_path+'graphstatplots/plot_ccs_'+savename+'.pdf')


    #fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax[1,1].plot(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,1].fill_between(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1) - 2 *all_ccws[lineidx,:,:].std(axis = 1), all_ccws[lineidx,:,:].mean(axis = 1) + 2 *all_ccws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,1].plot(degbins[np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,1].lines[-1].get_color())
        ax[1,1].plot(degbins[int(np.maximum(0,np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[1,1].lines[-1].get_color())
        ax[1,1].plot(degbins[:-1],true_ccws[lineidx,:], linestyle = '--', color = ax[1,1].lines[-1].get_color())
    ax[1,1].legend()
    ax[1,1].set_ylabel(f'Number of nodes')
    ax[1,1].set_xlabel(f'Weighted clustering coefficient')
    ymax = all_ccws.max()
    ax[1,1].set_ylim(-0.1*ymax,1.1*ymax)
    #plt.savefig(base_path+'graphstatplots/plot_ccws_'+savename+'.pdf')


    #fig,ax = plt.subplots()
    all_spls *= num_points
    true_spls *= num_points
    for lineidx in range(1,num_lines): 
        ax[1,2].plot(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,2].fill_between(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1) - 2 *all_spls[lineidx,:,:].std(axis = 1), all_spls[lineidx,:,:].mean(axis = 1) + 2 *all_spls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,2].plot(splbins[np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,2].lines[-1].get_color())
        ax[1,2].plot(splbins[int(np.maximum(0,np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 4,color = ax[1,2].lines[-1].get_color())
        ax[1,2].plot(splbins[:-1], true_spls[lineidx,:], linestyle = '--', color = ax[1,2].lines[-1].get_color())
    ax[1,2].legend()
    ax[1,2].set_ylabel(f'Number of node pairs')
    ax[1,2].set_xlabel(f'Shortest Path Length')
    ax[1,2].set_xlim(1,splbins[np.where(all_spls.mean(axis=2).mean(axis = 0)>0)[0][-1]])
    #plt.savefig(base_path+'graphstatplots/plot_spl_'+savename+'.pdf')

    ax = enumerate_subplots(ax,fontsize = 20)
    plt.savefig(base_path+'graphstatplots/jointplot_'+savename+'.pdf')
    # fig,ax = plt.subplots()
    # for lineidx in range(1,num_lines):
    #     ax.plot(binstoplot(truecorrbins), plink[lineidx,:], label = f'dens={thesedens[lineidx]}')
    # ax.legend()
    # ax.set_ylabel('Link probability')
    # ax.set_xlabel('True correlation')
    # plt.savefig(base_path+'graphstatplots/plot_linkprob_'+savename+'.pdf')
# %%
# fig,ax = plt.subplots()
# ax.plot(denslist[:all_lls.shape[0]], fdr(all_lls,true_lls[:all_lls.shape[0]]).mean(axis=1))
# ax.fill_between(denslist[:all_lls.shape[0]], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.025) for idens in range(all_lls.shape[0])], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.975) for idens in range(all_lls.shape[0])], alpha = 0.4)
# ax.set_ylabel(f'False discovery rate')
# ax.set_xlabel(f'Density')
# ax.set_ylim(0,1)
#plt.savefig(base_path+'graphstatplots/plot_fdr_'+savename+'.pdf')



# %%
[quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.025) for idens in range(all_lls.shape[0])], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.975) for idens in range(all_lls.shape[0])]
# %%
all_degs[2,:,:].mean(axis=1)

# %%
thisnam = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')][0]
biksg = myload(thisnam)
biksg.shape
# %%
get_adj(biksg,0.01,weighted=False).sum()/(num_points*(num_points-1))
# %%
biksg[np.triu(np.ones_like(biksg), k = 1).T != 0] = biksg[np.triu(np.ones_like(biksg), k = 1) != 0]

# %%
for graphstatname in find(f'graphstats_*', base_path+'real/'):
    # all_dens, all_degs, all_lls, all_ccs,all_ccws, all_spls, plink,all_llquant1,all_llquant2=myload(graphstatname)
    all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = myload(graphstatname)
    allstats = [all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2]
    for istat,statmat in enumerate(allstats):
        if len(statmat.shape) == 1:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,3))
        elif len(statmat.shape) == 2:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,1,3))
        elif len(statmat.shape) == 3:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,1,1,3))
        else:
            raise ValueError(f'shape {statmat.shape}?!')
    all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = allstats
    savename = graphstatname.split('graphstats_',1)[1][:-4]#+'_size3'
    adjust_fontsize(2)
    var_name, corr_method, typ = savename.split('_',3)[:3]
    if (corr_method != 'pearson') and (corr_method != 'BI-KSG'):
        print('Uses degbins and not degbinssparse, degbinsw.')
        continue
    # weighted = np.bool_(graphstatname.split(typ+'_w',1)[1].split('_',1)[0])
    n_lat = np.int64(savename.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    # kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    # if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
    #     cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    # else:
    #     cov = kernel(spherical2cartesian(lon,lat))
    if corr_method == 'BI-KSG':
        ranks = True
    else:
        ranks = False
    #exec(open("calc_true.py").read())
    #true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
    # try:
    #     true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2 = myload(base_path+f'truestats/truegraphstats_matern{nu}_{len_scale}_{typ}_{len(ks)}_ranks{ranks}.txt')
    # except:
    #     print(graphstatname, ' no true stats available.')
    #     continue
    num_lines = all_degs.shape[0]
    thesedens = np.round(all_dens.mean(axis=1),3)
    all_degs *= num_points
    all_degws *= num_points
    all_ccs *= num_points
    all_ccws *= num_points
    # true_degs *= num_points
    # true_degws *= num_points
    # true_ccs *= num_points
    # true_ccws *= num_points
    fig,ax = plt.subplots(2,3, figsize=(3*onefigsize[0],2*onefigsize[1]))
    for lineidx in range(1,num_lines):    
        ax[0,0].plot(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,0].fill_between(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1) - 2 *all_degs[lineidx,:,:].std(axis = 1), all_degs[lineidx,:,:].mean(axis = 1) + 2 *all_degs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,0].plot(degbinssparse[np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[0,0].lines[-1].get_color())
        ax[0,0].plot(degbinssparse[int(np.maximum(0,np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[0,0].lines[-1].get_color())
        # ax[0,0].plot(degbins[:-1],true_degs[lineidx,:], linestyle = '--', color = ax[0,0].lines[-1].get_color())
    ax[0,0].legend()
    ax[0,0].set_ylabel(f'Number of nodes')
    ax[0,0].set_xlabel(f'Normalized degree')
    ax[0,0].set_xlim(0,0.25)
    # if not os.path.exists(base_path+'real/graphstatplots/'):
    #     os.mkdir(base_path+'real/graphstatplots/')
    # plt.savefig(base_path+'real/graphstatplots/plot_deg_'+savename+'.pdf')

    for lineidx in range(1,num_lines):    
        ax[1,0].plot(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,0].fill_between(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1) - 2 *all_degws[lineidx,:,:].std(axis = 1), all_degws[lineidx,:,:].mean(axis = 1) + 2 *all_degws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,0].plot(degbinsw[np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,0].lines[-1].get_color())
        ax[1,0].plot(degbinsw[int(np.maximum(0,np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[1,0].lines[-1].get_color())
        # ax[1,0].plot(degbins[:-1],true_degws[lineidx,:], linestyle = '--', color = ax[1,0].lines[-1].get_color())
    ax[1,0].legend()
    ax[1,0].set_ylabel(f'Number of nodes')
    ax[1,0].set_xlabel(f'Normalized weighted degree')
    ax[1,0].set_xlim(0,0.1)
    #plt.savefig(base_path+'real/graphstatplots/plot_degw_'+savename+'.pdf')


    for lineidx in range(1,num_lines):
        ax[0,2].plot(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,2].fill_between(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1) - 2 *all_lls[lineidx,:,:].std(axis = 1), all_lls[lineidx,:,:].mean(axis = 1) + 2 *all_lls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,2].plot(all_llquant1[lineidx,:].mean(),0,'x',color = ax[0,2].lines[-1].get_color())
        ax[0,2].plot(all_llquant2[lineidx,:].mean(),0,'o',color = ax[0,2].lines[-1].get_color())
        # upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
        ax[0,2].plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax[0,2].lines[-1].get_color())
    ax[0,2].legend()
    ax[0,2].set_ylabel(f'Number of links')
    ax[0,2].set_xlabel(f'Distance (in radians)')
    ymax = all_lls.max()
    ax[0,2].set_ylim(-0.1*ymax,1.1*ymax)
    ax[0,2].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax[0,2].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax[0,2].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    #plt.savefig(base_path+'real/graphstatplots/plot_ll_'+savename+'.pdf')

    # fig,ax = plt.subplots()
    # ax.plot(denslist[:all_lls.shape[0]], fdr(all_lls,true_lls[:all_lls.shape[0]]).mean(axis=1))
    # ax.fill_between(denslist[:all_lls.shape[0]], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.025) for idens in range(all_lls.shape[0])], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.975) for idens in range(all_lls.shape[0])], alpha = 0.4)
    # ax.set_ylabel(f'False discovery rate')
    # ax.set_xlabel(f'Density')
    # ax.set_ylim(0,1)
    # plt.savefig(base_path+'real/graphstatplots/plot_fdr_'+savename+'.pdf')

    # fig,ax = plt.subplots()
    # for lineidx in range(1,num_lines):
    #     ax.plot(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
    #     ax.fill_between(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1) - 2 *all_llws[lineidx,:,:].std(axis = 1), all_llws[lineidx,:,:].mean(axis = 1) + 2 *all_llws[lineidx,:,:].std(axis = 1), alpha = 0.4)
    #     ax.plot(all_llwquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
    #     ax.plot(all_llwquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
    #     # upper_ll = llbins[np.where(true_llws[lineidx,:]>0)[0][-1]]
    #     ax.plot((upper_ll,upper_ll), (0,100000), linestyle = '--', color = ax.lines[-1].get_color())
    # ax.legend()
    # ax.set_ylabel(f'Edge weight')
    # ax.set_xlabel(f'Distance (in radians)')
    # ymax = all_llws.max()
    # ax.set_ylim(-0.1*ymax,1.1*ymax)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    # #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    # plt.savefig(base_path+'real/graphstatplots/plot_llw_'+savename+'.pdf')


    for lineidx in range(1,num_lines):    
        ax[1,1].plot(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,1].fill_between(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1) - 2 *all_ccws[lineidx,:,:].std(axis = 1), all_ccws[lineidx,:,:].mean(axis = 1) + 2 *all_ccws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,1].plot(degbins[np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,1].lines[-1].get_color())
        ax[1,1].plot(degbins[int(np.maximum(0,np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[1,1].lines[-1].get_color())
        # ax[1,1].plot(degbins[:-1],true_ccws[lineidx,:], linestyle = '--', color = ax[1,1].lines[-1].get_color())
    ax[1,1].legend()
    ax[1,1].set_ylabel(f'Number of nodes')
    ax[1,1].set_xlabel(f'Weighted clustering coefficient')
    ymax = all_ccws.max()
    ax[1,1].set_ylim(-0.1*ymax,1.1*ymax)

    for lineidx in range(1,num_lines):    
        ax[0,1].plot(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[0,1].fill_between(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1) - 2 *all_ccs[lineidx,:,:].std(axis = 1), all_ccs[lineidx,:,:].mean(axis = 1) + 2 *all_ccs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[0,1].plot(degbins[np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[0,1].lines[-1].get_color())
        ax[0,1].plot(degbins[int(np.maximum(0,np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax[0,1].lines[-1].get_color())
        # ax[0,1].plot(degbins[:-1],true_ccs[lineidx,:], linestyle = '--', color = ax[0,1].lines[-1].get_color())
    ax[0,1].legend()
    ax[0,1].set_ylabel(f'Number of nodes')
    ax[0,1].set_xlabel(f'Clustering coefficient')
    ymax = all_ccs.max()
    ax[0,1].set_ylim(-0.1*ymax,1.1*ymax)

    all_spls *= num_points
    # true_spls *= num_points
    for lineidx in range(1,num_lines): 
        ax[1,2].plot(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax[1,2].fill_between(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1) - 2 *all_spls[lineidx,:,:].std(axis = 1), all_spls[lineidx,:,:].mean(axis = 1) + 2 *all_spls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax[1,2].plot(splbins[np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax[1,2].lines[-1].get_color())
        ax[1,2].plot(splbins[int(np.maximum(0,np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 4,color = ax[1,2].lines[-1].get_color())
        # ax[1,2].plot(splbins[:-1], true_spls[lineidx,:], linestyle = '--', color = ax[1,2].lines[-1].get_color())
    ax[1,2].legend()
    ax[1,2].set_ylabel(f'Number of node pairs')
    ax[1,2].set_xlabel(f'Shortest Path Length')
    ax[1,2].set_xlim(1,splbins[np.where(all_spls.mean(axis=2).mean(axis = 0)>0)[0][-1]])
    
    ax = enumerate_subplots(ax,fontsize = 20)
    plt.savefig(base_path+'real/graphstatplots/jointplot_'+savename+'.pdf')
    # plt.savefig(base_path+'real/graphstatplots/plot_spl_'+savename+'.pdf')



# %%
for graphstatname in find(f'graphstats_*', base_path+'real/'):
    # all_dens, all_degs, all_lls, all_ccs,all_ccws, all_spls, plink,all_llquant1,all_llquant2=myload(graphstatname)
    all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = myload(graphstatname)
    allstats = [all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2]
    for istat,statmat in enumerate(allstats):
        if len(statmat.shape) == 1:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,3))
        elif len(statmat.shape) == 2:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,1,3))
        elif len(statmat.shape) == 3:
            allstats[istat] = np.tile(statmat[...,np.newaxis],(1,1,1,3))
        else:
            raise ValueError(f'shape {statmat.shape}?!')
    all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = allstats
    savename = graphstatname.split('graphstats_',1)[1][:-4]#+'_size3'
    adjust_fontsize(2)
    var_name, corr_method, typ = savename.split('_',3)[:3]
    if (corr_method != 'pearson') and (corr_method != 'BI-KSG'):
        print('Uses degbins and not degbinssparse, degbinsw.')
        continue
    # weighted = np.bool_(graphstatname.split(typ+'_w',1)[1].split('_',1)[0])
    n_lat = np.int64(savename.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    # kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    # if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
    #     cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    # else:
    #     cov = kernel(spherical2cartesian(lon,lat))
    if corr_method == 'BI-KSG':
        ranks = True
    else:
        ranks = False
    #exec(open("calc_true.py").read())
    #true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad
    # try:
    #     true_dens,true_densw, true_degs, true_lls, true_ccs,true_degws, true_llws,true_ccws, true_spls,true_llquant1,true_llquant2,true_llwquant1,true_llwquant2 = myload(base_path+f'truestats/truegraphstats_matern{nu}_{len_scale}_{typ}_{len(ks)}_ranks{ranks}.txt')
    # except:
    #     print(graphstatname, ' no true stats available.')
    #     continue
    num_lines = all_degs.shape[0]
    thesedens = np.round(all_dens.mean(axis=1),3)
    all_degs *= num_points
    all_degws *= num_points
    all_ccs *= num_points
    all_ccws *= num_points
    # true_degs *= num_points
    # true_degws *= num_points
    # true_ccs *= num_points
    # true_ccws *= num_points
    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax.plot(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(degbinssparse[:-1], all_degs[lineidx,:,:].mean(axis = 1) - 2 *all_degs[lineidx,:,:].std(axis = 1), all_degs[lineidx,:,:].mean(axis = 1) + 2 *all_degs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(degbinssparse[np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax.lines[-1].get_color())
        ax.plot(degbinssparse[int(np.maximum(0,np.where(all_degs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax.lines[-1].get_color())
        # ax.plot(degbins[:-1],true_degs[lineidx,:], linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of nodes')
    ax.set_xlabel(f'Normalized degree')
    ax.set_xlim(0,0.25)
    if not os.path.exists(base_path+'real/graphstatplots/'):
        os.mkdir(base_path+'real/graphstatplots/')
    plt.savefig(base_path+'real/graphstatplots/plot_deg_'+savename+'.pdf')

    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax.plot(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(degbinsw[:-1], all_degws[lineidx,:,:].mean(axis = 1) - 2 *all_degws[lineidx,:,:].std(axis = 1), all_degws[lineidx,:,:].mean(axis = 1) + 2 *all_degws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(degbinsw[np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax.lines[-1].get_color())
        ax.plot(degbinsw[int(np.maximum(0,np.where(all_degws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax.lines[-1].get_color())
        # ax.plot(degbins[:-1],true_degws[lineidx,:], linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of nodes')
    ax.set_xlabel(f'Normalized weighted degree')
    ax.set_xlim(0,0.1)
    plt.savefig(base_path+'real/graphstatplots/plot_degw_'+savename+'.pdf')


    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):
        ax.plot(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(llbins[:-1], all_lls[lineidx,:,:].mean(axis = 1) - 2 *all_lls[lineidx,:,:].std(axis = 1), all_lls[lineidx,:,:].mean(axis = 1) + 2 *all_lls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(all_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
        ax.plot(all_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
        # upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
        ax.plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of links')
    ax.set_xlabel(f'Distance (in radians)')
    ymax = all_lls.max()
    ax.set_ylim(-0.1*ymax,1.1*ymax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plt.savefig(base_path+'real/graphstatplots/plot_ll_'+savename+'.pdf')

    # fig,ax = plt.subplots()
    # ax.plot(denslist[:all_lls.shape[0]], fdr(all_lls,true_lls[:all_lls.shape[0]]).mean(axis=1))
    # ax.fill_between(denslist[:all_lls.shape[0]], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.025) for idens in range(all_lls.shape[0])], [quantile(fdr(all_lls,true_lls[:all_lls.shape[0]])[idens,:],alpha=0.975) for idens in range(all_lls.shape[0])], alpha = 0.4)
    # ax.set_ylabel(f'False discovery rate')
    # ax.set_xlabel(f'Density')
    # ax.set_ylim(0,1)
    # plt.savefig(base_path+'real/graphstatplots/plot_fdr_'+savename+'.pdf')

    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):
        ax.plot(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(llbins[:-1], all_llws[lineidx,:,:].mean(axis = 1) - 2 *all_llws[lineidx,:,:].std(axis = 1), all_llws[lineidx,:,:].mean(axis = 1) + 2 *all_llws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(all_llwquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
        ax.plot(all_llwquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
        # upper_ll = llbins[np.where(true_llws[lineidx,:]>0)[0][-1]]
        ax.plot((upper_ll,upper_ll), (0,100000), linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Edge weight')
    ax.set_xlabel(f'Distance (in radians)')
    ymax = all_llws.max()
    ax.set_ylim(-0.1*ymax,1.1*ymax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plt.savefig(base_path+'real/graphstatplots/plot_llw_'+savename+'.pdf')


    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax.plot(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(degbins[:-1], all_ccws[lineidx,:,:].mean(axis = 1) - 2 *all_ccws[lineidx,:,:].std(axis = 1), all_ccws[lineidx,:,:].mean(axis = 1) + 2 *all_ccws[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(degbins[np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax.lines[-1].get_color())
        ax.plot(degbins[int(np.maximum(0,np.where(all_ccws[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax.lines[-1].get_color())
        # ax.plot(degbins[:-1],true_ccws[lineidx,:], linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of nodes')
    ax.set_xlabel(f'Weighted clustering coefficient')
    ymax = all_ccws.max()
    ax.set_ylim(-0.1*ymax,1.1*ymax)
    plt.savefig(base_path+'real/graphstatplots/plot_ccws_'+savename+'.pdf')

    fig,ax = plt.subplots()
    for lineidx in range(1,num_lines):    
        ax.plot(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(degbins[:-1], all_ccs[lineidx,:,:].mean(axis = 1) - 2 *all_ccs[lineidx,:,:].std(axis = 1), all_ccs[lineidx,:,:].mean(axis = 1) + 2 *all_ccs[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(degbins[np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax.lines[-1].get_color())
        ax.plot(degbins[int(np.maximum(0,np.where(all_ccs[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 5,color = ax.lines[-1].get_color())
        # ax.plot(degbins[:-1],true_ccs[lineidx,:], linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of nodes')
    ax.set_xlabel(f'Clustering coefficient')
    ymax = all_ccs.max()
    ax.set_ylim(-0.1*ymax,1.1*ymax)
    plt.savefig(base_path+'real/graphstatplots/plot_ccs_'+savename+'.pdf')


    fig,ax = plt.subplots()
    all_spls *= num_points
    # true_spls *= num_points
    for lineidx in range(1,num_lines): 
        ax.plot(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1), label = f'dens={thesedens[lineidx]}')
        ax.fill_between(splbins[:-1], all_spls[lineidx,:,:].mean(axis = 1) - 2 *all_spls[lineidx,:,:].std(axis = 1), all_spls[lineidx,:,:].mean(axis = 1) + 2 *all_spls[lineidx,:,:].std(axis = 1), alpha = 0.4)
        ax.plot(splbins[np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][-1]],0,marker = 4,color = ax.lines[-1].get_color())
        ax.plot(splbins[int(np.maximum(0,np.where(all_spls[lineidx,:,:].mean(axis=1)>0)[0][0]-1))],0,marker = 4,color = ax.lines[-1].get_color())
        # ax.plot(splbins[:-1], true_spls[lineidx,:], linestyle = '--', color = ax.lines[-1].get_color())
    ax.legend()
    ax.set_ylabel(f'Number of node pairs')
    ax.set_xlabel(f'Shortest Path Length')
    ax.set_xlim(1,splbins[np.where(all_spls.mean(axis=2).mean(axis = 0)>0)[0][-1]])
    plt.savefig(base_path+'real/graphstatplots/plot_spl_'+savename+'.pdf')


    # fig,ax = plt.subplots()
    # for lineidx in range(1,num_lines):
    #     ax.plot(binstoplot(truecorrbins), plink[lineidx,:], label = f'dens={thesedens[lineidx]}')
    # ax.legend()
    # ax.set_ylabel('Link probability')
    # ax.set_xlabel('True correlation')
    # plt.savefig(base_path+'real/graphstatplots/plot_linkprob_'+savename+'.pdf')
    
# %%
savename = find(f'graphstats_*', base_path)[0].split('graphstats_',1)[1][:-4]
a,b,c=savename.split('_',3)[:3]
a,b,c
# %%
globals()['all_dens']

# %%
x = np.linspace(0,np.pi,100)
y = np.sin(x)
adjust_fontsize(2)
fig,a= plt.subplots()
a.plot(x,y)
a.get_window_extent()
# %%
onefigsize[1]*16/3.25