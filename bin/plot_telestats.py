# %%
import networkx as nx
import os
import numpy as np
import tueplots
import xarray as xr
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()
curr_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='spearman'
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True


ar = 0
ar2 = None
var = 10

num_runs = 2#30
n_time = 500
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5
len_scale = 0.1

denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]
#robust_tolerance = 0.5

filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'


#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99
robust_tolerance = 0.2

if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')

# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points,truecorrbins

# %%
#all_dens,all_tele1,all_tele2, all_robusttele2, all_mad, all_shufflemad = myload(base_path+ 'bundlestats/'+outfilename)
#f'telestats_part{irun}_{distrib}_{corr_method}_{typ}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}_{robust_tolerance}_{eps2}.txt'
#all_dens,all_tele1,all_tele2, all_robusttele2, all_mad, all_shufflemad = myload(find('telestats_part0_*',base_path+'bundlestats/')[2])

#%%
# fig,ax = plt.subplots()
# ax.plot(denslist,all_mad/all_shufflemad.mean(axis=1))
# ax.set_xlabel('Density')
# ax.set_ylabel('MAD / shuffled MAD')
mad_dict = {}
for filename in find(f'alltelestats_*time{n_time}*',base_path+'bundlestats/'):
    try:
        all_dens,all_tele1,all_tele2, all_robusttele2,all_robusttele2raw,all_mad, all_shufflemad = myload(filename) #all_robusttele2raw, 
    except:
        print(filename, 'wrong amount of stats.')
        continue
    savename = filename.split('alltelestats_',1)[1][:-4]# + 'size3'
    adjust_fontsize(2)
    try:
        all_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = myload(find('allothertelestats_'+savename+'*',base_path+'bundlestats/')[0])
    except:
        print(savename, 'other tele not existent.')
    nu = np.float64(filename.split('nu',1)[1].split('_',1)[0])
    len_scale = filename.split('len',1)[1].split('_',1)[0]
    corr_method = filename.split('igrf_',1)[1].split('_',1)[0]
    typ = filename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = filename.split('_w',1)[1].split('_',1)[0]
    robust_tolerance = filename.split('_tol',1)[1].split('_',1)[0]
    ranks = False
    if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
        ranks = True
    # weighted = np.bool_(filename.split(typ+'_w',1)[1].split('_',1)[0])
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
    truestats_file = f'truestats/truetelestats_matern{nu}_{len_scale}_{typ}_{len(ks)}_tol{robust_tolerance}_eps{eps2}_w{w}_ranks{ranks}.txt' #true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    try:
        true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    except:
        print(truestats_file, 'not existent.')
        continue
    # if not os.path.exists(base_path + truestats_file):
    #     exec(open("calc_true.py").read())
    #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw = true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw
    # else:
    #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    fig,ax = plt.subplots()
    pts = np.sum(dists <= 2 * dist_equator, axis = 1)
    ax.plot((denslist[0],denslist[-1]),(np.pi,np.pi), color = 'black')
    ax.plot((denslist[0],denslist[-1]),(0,0), color = 'black')
    ax.plot(denslist, all_tele1.mean(axis = 1), label = f'single link')# +- {np.round(pts.std(),1)}')
    #ax.fill_between(denslist, all_tele1.mean(axis = 1) - 2 *all_tele1.std(axis = 1), all_tele1.mean(axis = 1) + 2 *all_tele1.std(axis = 1), alpha = 0.4)
    ax.fill_between(denslist, [quantile(all_tele1[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_tele1[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_tele1, linestyle ='--',color = ax.lines[-1].get_color())#ax.plot(denslist, all_llquant1.mean(axis = 1), label = f'{alpha1} quantile')
    ax.plot(denslist, all_tele2.mean(axis = 1), label = f'1 to many')# +- {np.round(pts.std(),1)}')
    ax.fill_between(denslist, [quantile(all_tele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_tele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_tele2, linestyle ='--',color = ax.lines[-1].get_color())
    ax.plot(denslist, all_robusttele2.mean(axis = 1), label = f'loc. w. mtm')
    ax.fill_between(denslist, [quantile(all_robusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_robusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_robusttele2, linestyle ='--',color = ax.lines[-1].get_color())
    ax.plot(denslist, all_robusttele2raw.mean(axis = 1), label = f'many to many')
    ax.fill_between(denslist, [quantile(all_robusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_robusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_robusttele2raw, linestyle ='--',color = ax.lines[-1].get_color())
    #ax.fill_between(denslist, all_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
    #ax.plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
    #ax.fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
    ax.legend()
    ax.set_ylabel(f'Distance (in radians)')
    ax.set_ylim(-0.1,np.pi+0.1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel('Density')
    if not os.path.exists(base_path+'teleplots/'):
        os.mkdir(base_path+'teleplots/')
    plt.savefig(base_path + f'teleplots/teleplot_{np.round(pts.mean(),1)}_' + savename+'.pdf')

    #fig,ax = plt.subplots()
    mad_quotient = all_mad/all_shufflemad.mean(axis=1)
    mad_dict[savename] = mad_quotient
    # ax.plot(denslist,mad_quotient.mean(axis=1))
    # ax.fill_between(denslist, mad_quotient.mean(axis = 1) - 2 *mad_quotient.std(axis = 1), mad_quotient.mean(axis = 1) + 2 *mad_quotient.std(axis = 1), alpha = 0.4)
    # ax.set_xlabel('Density')
    # ax.set_ylabel('MAD / shuffled MAD')
    # plt.savefig(base_path + f'teleplots/madplot_{np.round(pts.mean(),1)}_' + savename+'.pdf')

# %%
#find('alltelestats_'+savename+'*',base_path+'bundlestats/')
# %%
for filename in find(f'allothertelestats_*time{n_time}',base_path+'bundlestats/'):
    print(filename)
    wrong_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = myload(filename)
    savename = filename.split('allothertelestats_',1)[1][:-4]
    sizechar = ''#'size3'
    adjust_fontsize(2)
    try:
        all_dens,all_tele1,_, all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad = myload(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0]) #all_robusttele2raw, 
        #all_dens,all_tele1, , all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad
    except:
        print(filename, 'wrong amount of stats.')
        continue
    nu = np.float64(filename.split('nu',1)[1].split('_',1)[0])
    len_scale = filename.split('len',1)[1].split('_',1)[0]
    corr_method = filename.split('igrf_',1)[1].split('_',1)[0]
    typ = filename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = filename.split('_w',1)[1].split('_',1)[0]
    robust_tolerance = filename.split('_tol',1)[1].split('_',1)[0]
    ranks = False
    # if nu != 0.5 or len_scale != '0.1' or w == 'False':
    #     continue
    if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
        ranks = True
    # weighted = np.bool_(filename.split(typ+'_w',1)[1].split('_',1)[0])
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
    
    truestats_file = f'truestats/truetelestats_matern{nu}_{len_scale}_{typ}_{len(ks)}_tol{robust_tolerance}_eps{eps2}_w{w}_ranks{ranks}.txt' #true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    # try:
    #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    # except:
    #     print(truestats_file, 'true stats do not exist.')
    #     continue

    # try:
    #     all_densunif,all_counttele2unif, all_countrobusttele2unif, all_countrobusttele2rawunif = myload(find('alluniformtelestats_'+savename+'*',base_path+'bundlestats/')[0])
    # except:
    #     print(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0], 'uniform stats do not exist.')
    #     continue
    # if not os.path.exists(base_path + truestats_file):
    #     exec(open("calc_true.py").read())
    #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw = true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw
    # else:
    #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
    
    print(f'TELE: {all_tele2.mean(axis=1)}, {all_robusttele2.mean(axis = 1)}')
    fig,ax = plt.subplots()
    pts = np.sum(dists <= 2 * dist_equator, axis = 1)
    ax.plot((denslist[0],denslist[-1]),(np.pi,np.pi), color = 'black')
    ax.plot((denslist[0],denslist[-1]),(0,0), color = 'black')
    ax.plot(denslist, all_tele1.mean(axis = 1), label = f'single link')# +- {np.round(pts.std(),1)}')
    ax.fill_between(denslist, [quantile(all_tele1[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele1[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_tele1, linestyle ='--',color = ax.lines[-1].get_color())#ax.plot(denslist, all_llquant1.mean(axis = 1), label = f'{alpha1} quantile')
    ax.plot(denslist, all_tele2.mean(axis = 1), label = f'1 to many')# +- {np.round(pts.std(),1)}')
    ax.fill_between(denslist, [quantile(all_tele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_tele2, linestyle ='--',color = ax.lines[-1].get_color())
    ax.plot(denslist, all_robusttele2.mean(axis = 1), label = f'loc. w. mtm')
    ax.fill_between(denslist, [quantile(all_robusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_robusttele2, linestyle ='--',color = ax.lines[-1].get_color())
    ax.plot(denslist, all_robusttele2raw.mean(axis = 1), label = f'many to many')
    ax.fill_between(denslist, [quantile(all_robusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
    ax.plot(denslist, true_robusttele2raw, linestyle ='--',color = ax.lines[-1].get_color())
    #ax.fill_between(denslist, all_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
    #ax.plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
    #ax.fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
    ax.legend()
    ax.set_ylabel(f'Distance (in radians)')
    ax.set_ylim(-0.1,np.pi+0.1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel('Density')
    if not os.path.exists(base_path+'teleplots/'):
        os.mkdir(base_path+'teleplots/')
    plt.savefig(base_path + f'teleplots/teleplot_{np.round(pts.mean(),1)}_' + savename+sizechar+'.pdf')


    '''mad_quotient = all_mad/all_shufflemad.mean(axis=1)
    if savename not in mad_dict:
        mad_dict[savename] = mad_quotient'''
    # fig,ax = plt.subplots()
    # ax.plot(denslist,mad_quotient.mean(axis=1))
    # ax.fill_between(denslist, mad_quotient.mean(axis = 1) - 2 *mad_quotient.std(axis = 1), mad_quotient.mean(axis = 1) + 2 *mad_quotient.std(axis = 1), alpha = 0.4)
    # ax.set_xlabel('Density')
    # ax.set_ylabel('MAD / shuffled MAD')
    # plt.savefig(base_path + f'teleplots/madplot_' + savename+'.pdf')

    #fdr_dict[savename] = all_fdr
    # fig,ax = plt.subplots()
    # ax.plot(denslist,all_fdr.mean(axis=1))
    # ax.fill_between(denslist, all_fdr.mean(axis = 1) - 2 *all_fdr.std(axis = 1), all_fdr.mean(axis = 1) + 2 *all_fdr.std(axis = 1), alpha = 0.4)
    # ax.set_xlabel('Density')
    # ax.set_ylabel('FDR')
    # plt.savefig(base_path + f'teleplots/fdrplot_' + savename+'.pdf')

    fig,ax = plt.subplots()
    pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
    ax.plot(denslist, all_counttele2.mean(axis = 1), label = f'1 to many', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
    ax.fill_between(denslist, [quantile(all_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
    #ax.plot(denslist, all_counttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
    #ax.fill_between(denslist, [quantile(all_counttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
    #ax.fill_between(denslist, all_counttele2.mean(axis = 1) - 2 *all_counttele2.std(axis = 1), all_counttele2.mean(axis = 1) + 2 *all_counttele2.std(axis = 1), alpha = 0.4, color = 'tab:orange')
    ax.plot(denslist, all_countrobusttele2.mean(axis = 1), label = f'loc. w. mtm', color = 'tab:green')
    ax.fill_between(denslist, [quantile(all_countrobusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
    #ax.plot(denslist, all_countrobusttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:green')
    #ax.fill_between(denslist, [quantile(all_countrobusttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
    ax.plot(denslist, all_countrobusttele2raw.mean(axis = 1), label = f'many to many', color = 'tab:red')
    ax.fill_between(denslist, [quantile(all_countrobusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
    #ax.plot(denslist, all_countrobusttele2rawunif.mean(axis = 1), linestyle = 'dotted', color = 'tab:red')
    #ax.fill_between(denslist, [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
    #ax.fill_between(denslist, altrue_robustcounttele2rawl_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
    #ax.plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
    #ax.fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
    ax.legend()
    ax.set_ylabel(f'Fraction of false links in bundles')
    #ax.set_ylim(-0.01,0.2)
    #ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel('Density')
    #ylim = 0.05
    #ax.set_ylim(1e-10,1)
    #ax.set_yscale('log')
    print(all_countrobusttele2.mean(axis=1))
    if not os.path.exists(base_path+'teleplots/'):
        os.mkdir(base_path+'teleplots/')
    plt.savefig(base_path + f'teleplots/countteleplot_{np.round(pts.mean(),1)}_' + savename+sizechar+'.pdf')
# %%
print(all_countrobusttele2.mean(axis=1))

# %%
fdr_dict = {}
for filename in find('allfdr_*',base_path+'bundlestats/'):
    print(filename)
    all_fdr = myload(filename)
    savename = filename.split('allfdr_',1)[1][:-4]
    fdr_dict[savename] = all_fdr

# %%
# plot mad and fdr jointly

corr_method0 = 'pearson'
typ0 = 'threshold'
w0 = 'True'
n_lat0 = 72
robust_tolerance0 = '0.2'
sortedmad_dict={}
sortedfdr_dict={}
for nu in [0.5,1.5]:
    for len_scale in ['0.1','0.2']:
        for savename in mad_dict:
            thisnu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
            thislen_scale = savename.split('len',1)[1].split('_',1)[0]
            if thisnu == nu and thislen_scale == len_scale:
                sortedmad_dict[savename] = mad_dict[savename]
        for savename in fdr_dict:
            thisnu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
            thislen_scale = savename.split('len',1)[1].split('_',1)[0]
            if thisnu == nu and thislen_scale == len_scale:
                sortedfdr_dict[savename] = fdr_dict[savename]

fig,ax = plt.subplots()
for savename in sortedmad_dict:
    nu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
    len_scale = savename.split('len',1)[1].split('_',1)[0]
    corr_method = savename.split('igrf_',1)[1].split('_',1)[0]
    typ = savename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = savename.split('_w',1)[1].split('_',1)[0]
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    robust_tolerance = savename.split('_tol',1)[1].split('_',1)[0]
    print(savename)
    if (corr_method != corr_method0) or (typ != typ0) or (w != w0) or (n_lat!= n_lat0) or (robust_tolerance != robust_tolerance0):
        print((corr_method != corr_method0), (typ != typ0), (w != w0), (n_lat!= n_lat0), (robust_tolerance != robust_tolerance0))
        continue
    mad_quotient = mad_dict[savename]
    ax.plot(denslist,mad_quotient.mean(axis=1), label = f'nu={nu}, l={len_scale}')
    ax.fill_between(denslist, [quantile(mad_quotient[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(mad_quotient[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
ax.set_xlabel('Density')
ax.set_ylabel('MAD / shuffled MAD')
ax.legend()
combined_name = f'{corr_method0}_{typ0}_w{w0}_fekete{n_lat0}_tol{robust_tolerance0}' + savename.split('ar',1)[1].split('_tol',1)[0]
plt.savefig(base_path + f'teleplots/combined_madplot_' + combined_name[0]+combined_name[1]+'.pdf')

fig,ax = plt.subplots()
for savename in sortedfdr_dict:
    nu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
    len_scale = savename.split('len',1)[1].split('_',1)[0]
    corr_method = savename.split('igrf_',1)[1].split('_',1)[0]
    typ = savename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = savename.split('_w',1)[1].split('_',1)[0]
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    robust_tolerance = savename.split('_tol',1)[1].split('_',1)[0]
    print(savename)
    if (corr_method != corr_method0) or (typ != typ0) or (w != w0) or (n_lat!= n_lat0) or (robust_tolerance != robust_tolerance0):
        print((corr_method != corr_method0), (typ != typ0), (w != w0), (n_lat!= n_lat0), (robust_tolerance != robust_tolerance0))
        continue
    fdr = sortedfdr_dict[savename]
    ax.plot(denslist,fdr.mean(axis=1), label = f'nu={nu}, l={len_scale}')
    ax.fill_between(denslist, [quantile(fdr[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(fdr[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
ax.set_xlabel('Density')
ax.set_ylabel('FDR')
ax.legend()
combined_name = f'{corr_method0}_{typ0}_w{w0}_fekete{n_lat0}_tol{robust_tolerance0}' + savename.split('ar',1)[1].split('_tol',1)[0]
plt.savefig(base_path + f'teleplots/combined_fdrplot_' + combined_name+'.pdf')

# density = 0.01
# adj = get_adj(cov,0.05,weighted = False)
# distadj = get_adj(5-dists, density,weighted=True)
# maxdist = 5-distadj[distadj!=0].min()
# adj[np.logical_and(adj != 0, dists > maxdist)].sum()

# plot fdr and mad jointly

fig, axs = plt.subplots(1,2,figsize=(2*onefigsize[0],onefigsize[1]))
for savename in sortedfdr_dict:
    nu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
    len_scale = savename.split('len',1)[1].split('_',1)[0]
    corr_method = savename.split('igrf_',1)[1].split('_',1)[0]
    typ = savename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = savename.split('_w',1)[1].split('_',1)[0]
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    robust_tolerance = savename.split('_tol',1)[1].split('_',1)[0]
    print(savename)
    if (corr_method != corr_method0) or (typ != typ0) or (w != w0) or (n_lat!= n_lat0) or (robust_tolerance != robust_tolerance0):
        print((corr_method != corr_method0), (typ != typ0), (w != w0), (n_lat!= n_lat0), (robust_tolerance != robust_tolerance0))
        continue
    fdr = sortedfdr_dict[savename]
    axs[0].plot(denslist,fdr.mean(axis=1), label = f'nu={nu}, l={len_scale}')
    axs[0].fill_between(denslist, [quantile(fdr[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(fdr[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
axs[0].set_xlabel('Density')
axs[0].set_ylabel('FDR')
axs[0].legend()

for savename in sortedmad_dict:
    nu = np.float64(savename.split('nu',1)[1].split('_',1)[0])
    len_scale = savename.split('len',1)[1].split('_',1)[0]
    corr_method = savename.split('igrf_',1)[1].split('_',1)[0]
    typ = savename.split(corr_method+'_',1)[1].split('_',1)[0]
    w = savename.split('_w',1)[1].split('_',1)[0]
    n_lat = np.int64(filename.split('fekete',1)[1].split('_',1)[0])
    robust_tolerance = savename.split('_tol',1)[1].split('_',1)[0]
    print(savename)
    if (corr_method != corr_method0) or (typ != typ0) or (w != w0) or (n_lat!= n_lat0) or (robust_tolerance != robust_tolerance0):
        print((corr_method != corr_method0), (typ != typ0), (w != w0), (n_lat!= n_lat0), (robust_tolerance != robust_tolerance0))
        continue
    mad_quotient = mad_dict[savename]
    axs[1].plot(denslist,mad_quotient.mean(axis=1), label = f'nu={nu}, l={len_scale}')
    axs[1].fill_between(denslist, [quantile(mad_quotient[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(mad_quotient[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
axs[1].set_xlabel('Density')
axs[1].set_ylabel('MAD / shuffled MAD')
axs[1].legend()
axs = enumerate_subplots(axs,fontsize = 16)
combined_name = f'{corr_method0}_{typ0}_w{w0}_fekete{n_lat0}_tol{robust_tolerance0}' + savename.split('ar',1)[1].split('_tol',1)[0]
plt.savefig(base_path + f'teleplots/joint_fdrmadplot_' + combined_name+'.pdf')

# %%
# plot teleplots jointly
corr_method = 'pearson'
typ = 'threshold'
for nu in [0.5,1.5]:
    for len_scale in [0.1,0.2]:
        fig,axs = plt.subplots(2,2,figsize=(2*onefigsize[0],2*onefigsize[1]))
        try:
            filename1 = find(f'alltelestats_*{corr_method}_{typ}*wFalse*nu{nu}_len{len_scale}*tol0.2*',base_path+'bundlestats/')[0]
            filename2 = find(f'allothertelestats_*{corr_method}_{typ}*wFalse*nu{nu}_len{len_scale}*tol0.2*',base_path+'bundlestats/')[0]
            filename3 = find(f'alltelestats_*{corr_method}_{typ}*wTrue*nu{nu}_len{len_scale}*tol0.5*',base_path+'bundlestats/')[0]
            filename4 = find(f'allothertelestats_*{corr_method}_{typ}*wTrue*nu{nu}_len{len_scale}*tol0.5*',base_path+'bundlestats/')[0]
        except:
            print('Some file does not exist for nu,len=',nu,len_scale,corr_method)
        
        print(filename2)
        wrong_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = myload(filename2)
        savename = filename2.split('allothertelestats_',1)[1][:-4]
        sizechar = ''#'size3'
        adjust_fontsize(2)
        try:
            all_dens,all_tele1,_, all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad = myload(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0]) #all_robusttele2raw, 
            #all_dens,all_tele1, , all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad
        except:
            print(filename2, 'wrong amount of stats.')
            continue
        w=False
        ranks = False
        # if nu != 0.5 or len_scale != '0.1' or w == 'False':
        #     continue
        if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
            ranks = True
        # weighted = np.bool_(filename2.split(typ+'_w',1)[1].split('_',1)[0])
        n_lat = np.int64(filename2.split('fekete',1)[1].split('_',1)[0])
        num_points = gridstep_to_numpoints(180/n_lat)
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
            cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
        else:
            cov = kernel(spherical2cartesian(lon,lat))
        
        truestats_file = f'truestats/truetelestats_matern{nu}_{len_scale}_{typ}_{len(ks)}_tol{robust_tolerance}_eps{eps2}_w{w}_ranks{ranks}.txt' #true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        try:
            true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        except:
            print(truestats_file, 'true stats do not exist.')
            continue

        # try:
        #     all_densunif,all_counttele2unif, all_countrobusttele2unif, all_countrobusttele2rawunif = myload(find('alluniformtelestats_'+savename+'*',base_path+'bundlestats/')[0])
        # except:
        #     print(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0], 'uniform stats do not exist.')
        #     continue
        # if not os.path.exists(base_path + truestats_file):
        #     exec(open("calc_true.py").read())
        #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw = true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw
        # else:
        #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        
        print(f'TELE: {all_tele2.mean(axis=1)}, {all_robusttele2.mean(axis = 1)}')
        
        pts = np.sum(dists <= 2 * dist_equator, axis = 1)
        axs[0,0].plot((denslist[0],denslist[-1]),(np.pi,np.pi), color = 'black')
        axs[0,0].plot((denslist[0],denslist[-1]),(0,0), color = 'black')
        axs[0,0].plot(denslist, all_tele1.mean(axis = 1), label = f'single link')# +- {np.round(pts.std(),1)}')
        axs[0,0].fill_between(denslist, [quantile(all_tele1[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele1[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[0,0].plot(denslist, true_tele1, linestyle ='--',color = axs[0,0].lines[-1].get_color())#axs[0,0].plot(denslist, all_llquant1.mean(axis = 1), label = f'{alpha1} quantile')
        axs[0,0].plot(denslist, all_tele2.mean(axis = 1), label = f'1 to many')# +- {np.round(pts.std(),1)}')
        axs[0,0].fill_between(denslist, [quantile(all_tele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[0,0].plot(denslist, true_tele2, linestyle ='--',color = axs[0,0].lines[-1].get_color())
        axs[0,0].plot(denslist, all_robusttele2.mean(axis = 1), label = f'loc. w. mtm')
        axs[0,0].fill_between(denslist, [quantile(all_robusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[0,0].plot(denslist, true_robusttele2, linestyle ='--',color = axs[0,0].lines[-1].get_color())
        axs[0,0].plot(denslist, all_robusttele2raw.mean(axis = 1), label = f'many to many')
        axs[0,0].fill_between(denslist, [quantile(all_robusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[0,0].plot(denslist, true_robusttele2raw, linestyle ='--',color = axs[0,0].lines[-1].get_color())
        #axs[0,0].fill_between(denslist, all_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
        #axs[0,0].plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
        #axs[0,0].fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
        axs[0,0].legend()
        axs[0,0].set_ylabel(f'Distance (in radians)')
        axs[0,0].set_ylim(-0.1,np.pi+0.1)
        axs[0,0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        #axs[0,0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        axs[0,0].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[0,0].set_xlabel('Density')

        pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
        axs[0,1].plot(denslist, all_counttele2.mean(axis = 1), label = f'1 to many', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
        axs[0,1].fill_between(denslist, [quantile(all_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
        #axs[0,1].plot(denslist, all_counttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
        #axs[0,1].fill_between(denslist, [quantile(all_counttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
        #axs[0,1].fill_between(denslist, all_counttele2.mean(axis = 1) - 2 *all_counttele2.std(axis = 1), all_counttele2.mean(axis = 1) + 2 *all_counttele2.std(axis = 1), alpha = 0.4, color = 'tab:orange')
        axs[0,1].plot(denslist, all_countrobusttele2.mean(axis = 1), label = f'loc. w. mtm', color = 'tab:green')
        axs[0,1].fill_between(denslist, [quantile(all_countrobusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
        #axs[0,1].plot(denslist, all_countrobusttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:green')
        #axs[0,1].fill_between(denslist, [quantile(all_countrobusttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
        axs[0,1].plot(denslist, all_countrobusttele2raw.mean(axis = 1), label = f'many to many', color = 'tab:red')
        axs[0,1].fill_between(denslist, [quantile(all_countrobusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
        #axs[0,1].plot(denslist, all_countrobusttele2rawunif.mean(axis = 1), linestyle = 'dotted', color = 'tab:red')
        #axs[0,1].fill_between(denslist, [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
        #axs[0,1].fill_between(denslist, altrue_robustcounttele2rawl_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
        #axs[0,1].plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
        #axs[0,1].fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
        axs[0,1].legend()
        axs[0,1].set_ylabel(f'Fraction of false links in bundles')
        #axs[0,1].set_ylim(-0.1,np.pi+0.1)
        #axs[0,1].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        #axs[0,1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        #axs[0,1].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[0,1].set_xlabel('Density')
        #ylim = 0.05
        #axs[0,1].set_ylim(1e-10,1)
        #axs[0,1].set_yscale('log')
        print(all_countrobusttele2.mean(axis=1))
        # if not os.path.exists(base_path+'teleplots/'):
        #     os.mkdir(base_path+'teleplots/')
        # plt.savefig(base_path + f'teleplots/countteleplot_{np.round(pts.mean(),1)}_' + savename+sizechar+'.pdf')

        print(filename4)
        wrong_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = myload(filename4)
        savename = filename4.split('allothertelestats_',1)[1][:-4]
        sizechar = ''#'size3'
        adjust_fontsize(2)
        try:
            all_dens,all_tele1,_, all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad = myload(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0]) #all_robusttele2raw, 
            #all_dens,all_tele1, , all_robusttele2, all_robusttele2raw, all_mad, all_shufflemad
        except:
            print(filename4, 'wrong amount of stats.')
            continue
        w= True
        ranks = False
        # if nu != 0.5 or len_scale != '0.1' or w == 'False':
        #     continue
        if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
            ranks = True
        # weighted = np.bool_(filename4.split(typ+'_w',1)[1].split('_',1)[0])
        n_lat = np.int64(filename4.split('fekete',1)[1].split('_',1)[0])
        num_points = gridstep_to_numpoints(180/n_lat)
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
            cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
        else:
            cov = kernel(spherical2cartesian(lon,lat))
        
        truestats_file = f'truestats/truetelestats_matern{nu}_{len_scale}_{typ}_{len(ks)}_tol{robust_tolerance}_eps{eps2}_w{w}_ranks{ranks}.txt' #true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        try:
            true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        except:
            print(truestats_file, 'true stats do not exist.')
            continue

        # try:
        #     all_densunif,all_counttele2unif, all_countrobusttele2unif, all_countrobusttele2rawunif = myload(find('alluniformtelestats_'+savename+'*',base_path+'bundlestats/')[0])
        # except:
        #     print(find('alltelestats_'+savename+'*',base_path+'bundlestats/')[0], 'uniform stats do not exist.')
        #     continue
        # if not os.path.exists(base_path + truestats_file):
        #     exec(open("calc_true.py").read())
        #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw = true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad, true_densw, true_llws, true_llwquant1,true_llwquant2,true_telew1,true_telew2,true_robusttelew2,true_robusttelew2raw,  true_madw, true_shufflemadw
        # else:
        #     true_dens, true_lls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_robusttele2raw, true_mad, true_shufflemad = myload(base_path+truestats_file)
        
        print(f'TELE: {all_tele2.mean(axis=1)}, {all_robusttele2.mean(axis = 1)}')
        
        pts = np.sum(dists <= 2 * dist_equator, axis = 1)
        axs[1,0].plot((denslist[0],denslist[-1]),(np.pi,np.pi), color = 'black')
        axs[1,0].plot((denslist[0],denslist[-1]),(0,0), color = 'black')
        axs[1,0].plot(denslist, all_tele1.mean(axis = 1), label = f'single link')# +- {np.round(pts.std(),1)}')
        axs[1,0].fill_between(denslist, [quantile(all_tele1[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele1[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[1,0].plot(denslist, true_tele1, linestyle ='--',color = axs[1,0].lines[-1].get_color())#axs[1,0].plot(denslist, all_llquant1.mean(axis = 1), label = f'{alpha1} quantile')
        axs[1,0].plot(denslist, all_tele2.mean(axis = 1), label = f'1 to many')# +- {np.round(pts.std(),1)}')
        axs[1,0].fill_between(denslist, [quantile(all_tele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_tele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[1,0].plot(denslist, true_tele2, linestyle ='--',color = axs[1,0].lines[-1].get_color())
        axs[1,0].plot(denslist, all_robusttele2.mean(axis = 1), label = f'loc. w. mtm')
        axs[1,0].fill_between(denslist, [quantile(all_robusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[1,0].plot(denslist, true_robusttele2, linestyle ='--',color = axs[1,0].lines[-1].get_color())
        axs[1,0].plot(denslist, all_robusttele2raw.mean(axis = 1), label = f'many to many')
        axs[1,0].fill_between(denslist, [quantile(all_robusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(all_robusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], alpha = 0.4)
        axs[1,0].plot(denslist, true_robusttele2raw, linestyle ='--',color = axs[1,0].lines[-1].get_color())
        #axs[1,0].fill_between(denslist, all_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
        #axs[1,0].plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
        #axs[1,0].fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
        axs[1,0].legend()
        axs[1,0].set_ylabel(f'Distance (in radians)')
        axs[1,0].set_ylim(-0.1,np.pi+0.1)
        axs[1,0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        #axs[1,0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        axs[1,0].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[1,0].set_xlabel('Density')

        pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
        axs[1,1].plot(denslist, all_counttele2.mean(axis = 1), label = f'1 to many', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
        axs[1,1].fill_between(denslist, [quantile(all_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
        #axs[1,1].plot(denslist, all_counttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
        #axs[1,1].fill_between(denslist, [quantile(all_counttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_counttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
        #axs[1,1].fill_between(denslist, all_counttele2.mean(axis = 1) - 2 *all_counttele2.std(axis = 1), all_counttele2.mean(axis = 1) + 2 *all_counttele2.std(axis = 1), alpha = 0.4, color = 'tab:orange')
        axs[1,1].plot(denslist, all_countrobusttele2.mean(axis = 1), label = f'loc. w. mtm', color = 'tab:green')
        axs[1,1].fill_between(denslist, [quantile(all_countrobusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
        #axs[1,1].plot(denslist, all_countrobusttele2unif.mean(axis = 1),linestyle = 'dotted', color = 'tab:green')
        #axs[1,1].fill_between(denslist, [quantile(all_countrobusttele2unif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2unif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
        axs[1,1].plot(denslist, all_countrobusttele2raw.mean(axis = 1), label = f'many to many', color = 'tab:red')
        axs[1,1].fill_between(denslist, [quantile(all_countrobusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
        #axs[1,1].plot(denslist, all_countrobusttele2rawunif.mean(axis = 1), linestyle = 'dotted', color = 'tab:red')
        #axs[1,1].fill_between(denslist, [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_countrobusttele2rawunif[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:red', alpha = 0.4)
        #axs[1,1].fill_between(denslist, altrue_robustcounttele2rawl_llquant1.mean(axis = 1) - 2 *all_llquant1.std(axis = 1), all_llquant1.mean(axis = 1) + 2 *all_llquant1.std(axis = 1), alpha = 0.4)
        #axs[1,1].plot(denslist, all_llquant2.mean(axis = 1), label = f'{alpha2} quantile')
        #axs[1,1].fill_between(denslist, all_llquant2.mean(axis = 1) - 2 *all_llquant2.std(axis = 1), all_llquant2.mean(axis = 1) + 2 *all_llquant2.std(axis = 1), alpha = 0.4)
        axs[1,1].legend()
        axs[1,1].set_ylabel(f'Fraction of false links in bundles')
        #axs[1,1].set_ylim(-0.1,np.pi+0.1)
        #axs[1,1].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        #axs[1,1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        #axs[1,1].yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[1,1].set_xlabel('Density')
        #ylim = 0.05
        #axs[1,1].set_ylim(1e-10,1)
        axs[1,1].set_yscale('log')
        print(all_countrobusttele2.mean(axis=1))
        # if not os.path.exists(base_path+'teleplots/'):
        #     os.mkdir(base_path+'teleplots/')
        axs = enumerate_subplots(axs,fontsize = 16)
        plt.savefig(base_path + f'teleplots/jointteleplot_{np.round(pts.mean(),1)}_' + savename+sizechar+'.pdf')
# %%
