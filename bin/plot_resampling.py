# %%
import os, fnmatch, pickle
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
#from multiprocessing import Pool
from collections import Counter
from sklearn.gaussian_process.kernels import Matern
from climnet.dataset_new import AnomalyDataset
start_time = time.time()
curr_time = time.time()
# from tueplots import bundles
# plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update({"figure.dpi": 300})


#irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
#print('Task: ', irun)
start_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

var_name = 't2m'
grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
corr_method='BI-KSG'
weighted = False
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True


ar = 0
ar2 = None
var = 10
if weighted:
    robust_tolerance = 0.5
else:
    robust_tolerance = 0.2

#num_surrogates = 10#0

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 1.5 # irrelevant
len_scale = 0.2 # irrelevant

denslist = [0.001,0.005,0.01,0.05,0.1]
ks = [6, 60, 300, 600,1200]


filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

# %%
#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

epsgeo = 0.05 # choose smartly?
n_rewire = 10 # * number of links
eps2 = 2 * dist_equator
eps3 = 3 * dist_equator
alpha1 = 0.95
alpha2 = 0.99


if ar2 is None:
    ar_coeff = ar * np.ones(num_points)
else:
    raise RuntimeError('Arbitrary AR values not implemented.')

# just to be detected as known variables in VSC
dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points = dists, numth, degbins, llbins, splbins, nbhds2, cov, all_degs, all_lls,all_ccs, all_ccws, all_spls, all_betw, all_eigc, all_dens, all_tele1, all_tele2, all_robusttele2, all_llquant1, all_llquant2, all_telequant, all_telequant2, all_mad, all_shufflemad, plink,dist_equator,num_points
#true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad = true_dens, true_degs, true_lls, true_ccs, true_spls, true_llquant1,true_llquant2,true_tele1,true_tele2,true_robusttele2, true_mad, true_shufflemad


geps = 0.05
rew = 10

# %%
all_geo_lls,all_geo_llquant1,all_geo_llquant2,all_geo_counttele2,all_geo_countrobusttele2,all_geo_countrobusttele2raw = [[] for _ in range(6)]
for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
    thesegeolls, thesellq1,thesellq2,thesecount2,thesecr2, thesecrc2=[[] for _ in range(6)]
    thesefiles = find(f'geostats_*_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}*',base_path+'geo/')[0]
    for irun in range(30):
        filename = find(f'geostats_run{irun}_*_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}*',base_path+'geo/')[0]
        one_geo_lls,one_geo_llquant1, one_geo_llquant2, one_geo_counttele2, one_geo_countrobusttele2, one_geo_countrobusttele2raw = myload(filename)
        if (one_geo_lls !=0).sum()!=0:
            thesegeolls.append(one_geo_lls)
            thesellq1.append(one_geo_llquant1)
            thesellq2.append(one_geo_llquant2)
            thesecount2.append(one_geo_counttele2)
            thesecr2.append(one_geo_countrobusttele2)
            thesecrc2.append(one_geo_countrobusttele2raw)    
    all_geo_lls.append(thesegeolls) # density x run x bins
    all_geo_llquant1.append(thesellq1)
    all_geo_llquant2.append(thesellq2)
    all_geo_counttele2.append(thesecount2)
    all_geo_countrobusttele2.append(thesecr2)
    all_geo_countrobusttele2raw.append(thesecrc2)

minruns=100
for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
    minruns = np.minimum(minruns, len(all_geo_lls[i]))

all_geo_lls_np = np.zeros((5,num_bins,minruns))
all_geo_counttele2_np = np.zeros((5,minruns))
for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
    for irun in range(minruns):
        try:
            all_geo_lls_np[i,:,irun] = all_geo_lls[i][irun]
            all_geo_counttele2_np[i,irun] = all_geo_counttele2[i][irun]
        except:
            print(i,irun)

all_geo_lls = all_geo_lls_np
all_geo_counttele2 = all_geo_counttele2_np
# all_geo_lls = np.array(all_geo_lls)
# all_geo_lls =  np.transpose(all_geo_lls, (0,2,1))
# all_geo_llquant1 = np.array(all_geo_llquant1)
# all_geo_llquant2= np.array(all_geo_llquant2)
# all_geo_counttele2= np.array(all_geo_counttele2)
# all_geo_countrobusttele2= np.array(all_geo_countrobusttele2)
# all_geo_countrobusttele2raw= np.array(all_geo_countrobusttele2raw)
# %%
for i, density in enumerate([0.001,0.005,0.01,0.05,0.1]):
    print(len(all_geo_lls[i][13]))
# %%
minruns
# %%
#len(all_geo_lls[3]), all_geo_lls[0][0].shape
find(f'geostats_*_dens{density}_geps{geps}_rew{rew}_nlat{n_lat}*',base_path)
# %%
num_links,one_lls,one_llquant1, one_llquant2, one_counttele2, one_countrobusttele2, one_countrobusttele2raw,all_iaaft_lls,all_iaaft_llquant1, all_iaaft_llquant2, all_iaaft_counttele2, all_iaaft_countrobusttele2, all_iaaft_countrobusttele2raw,all_boot_lls,all_boot_llquant1, all_boot_llquant2, all_boot_counttele2, all_boot_countrobusttele2, all_boot_countrobusttele2raw = myload(find('allresampling_*', base_path)[0])

lineidx = 3
print(f'dens={num_links[lineidx]/(2*num_points*(num_points-1))}')

fig,ax = plt.subplots()
ax.plot(llbins[:-1], one_lls[lineidx,:], label = 't2m',color='black')
ax.plot(one_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(one_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())

ax.plot(llbins[:-1], np.array(all_geo_lls[lineidx]).mean(axis = 1), label = 'Geomodel 2', color = 'tab:blue')
ax.fill_between(llbins[:-1], np.array(all_geo_lls[lineidx]).mean(axis = 1) - 2 *np.array(all_geo_lls[lineidx]).std(axis = 1), np.array(all_geo_lls[lineidx]).mean(axis = 1) + 2 *np.array(all_geo_lls[lineidx]).std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(np.mean(all_geo_llquant1[lineidx]),0,'x',color = ax.lines[-1].get_color())
ax.plot(np.mean(all_geo_llquant2[lineidx]),0,'o',color = ax.lines[-1].get_color())

ax.plot(llbins[:-1], all_iaaft_lls[lineidx,:,:].mean(axis = 1), label = 'IAAFT', color = 'tab:orange')
ax.fill_between(llbins[:-1], all_iaaft_lls[lineidx,:,:].mean(axis = 1) - 2 *all_iaaft_lls[lineidx,:,:].std(axis = 1), all_iaaft_lls[lineidx,:,:].mean(axis = 1) + 2 *all_iaaft_lls[lineidx,:,:].std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(all_iaaft_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(all_iaaft_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())

ax.plot(llbins[:-1], all_boot_lls[lineidx,:,:].mean(axis = 1), label = 'Bootstrap',color='tab:green')
ax.fill_between(llbins[:-1], all_boot_lls[lineidx,:,:].mean(axis = 1) - 2 *all_boot_lls[lineidx,:,:].std(axis = 1), all_boot_lls[lineidx,:,:].mean(axis = 1) + 2 *all_boot_lls[lineidx,:,:].std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(all_boot_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(all_boot_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
#upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
#ax.plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax.lines[-1].get_color())
ax.legend()
ax.set_ylabel(f'Number of links')
ax.set_xlabel(f'Distance (in radians)')
ymax = all_lls.max()
#ax.set_ylim(-0.1*ymax,1.1*ymax)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
#ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.savefig(base_path+'plot_llresampl_full_'+f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}'+'.pdf')

# %%
linksperbin = np.zeros(len(llbins)-1)
for ibin in range(len(llbins)-1):
    # how many links in bin
    linksperbin[ibin] = np.logical_and(llbins[ibin] <= dists, dists < llbins[ibin+1]).sum()

# %%
for ibin in range(len(llbins)-1):
    one_lls[:,ibin] = one_lls[:,ibin] / linksperbin[ibin]
    all_geo_lls[:,ibin] = all_geo_lls[:,ibin] / linksperbin[ibin]
    all_iaaft_lls[:,ibin] = all_iaaft_lls[:,ibin] / linksperbin[ibin]
    all_boot_lls[:,ibin] = all_boot_lls[:,ibin] / linksperbin[ibin]

# %%

fig,ax = plt.subplots()
ax.plot(llbins[1:-1], one_lls[lineidx,1:], label = 't2m',color='black')
ax.plot(one_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(one_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())


ax.plot(llbins[1:-1], all_geo_lls[lineidx,1:,:].mean(axis = 1), label = 'Geomodel 2', color = 'tab:blue')
ax.fill_between(llbins[1:-1], all_geo_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_geo_lls[lineidx,1:,:].std(axis = 1), all_geo_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_geo_lls[lineidx,1:,:].std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(np.mean(all_geo_llquant1[lineidx]),0,'x',color = ax.lines[-1].get_color())
ax.plot(np.mean(all_geo_llquant2[lineidx]),0,'o',color = ax.lines[-1].get_color())

ax.plot(llbins[1:-1], all_iaaft_lls[lineidx,1:,:].mean(axis = 1), label = 'IAAFT', color = 'tab:orange')
ax.fill_between(llbins[1:-1], all_iaaft_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_iaaft_lls[lineidx,1:,:].std(axis = 1), all_iaaft_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_iaaft_lls[lineidx,1:,:].std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(all_iaaft_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(all_iaaft_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())

ax.plot(llbins[1:-1], all_boot_lls[lineidx,1:,:].mean(axis = 1), label = 'Bootstrap',color='tab:green')
ax.fill_between(llbins[1:-1], all_boot_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_boot_lls[lineidx,1:,:].std(axis = 1), all_boot_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_boot_lls[lineidx,1:,:].std(axis = 1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.plot(all_boot_llquant1[lineidx,:].mean(),0,'x',color = ax.lines[-1].get_color())
ax.plot(all_boot_llquant2[lineidx,:].mean(),0,'o',color = ax.lines[-1].get_color())
#upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
#ax.plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = ax.lines[-1].get_color())
ax.legend()
ax.set_ylabel(f'Fraction of links')
ax.set_xlabel(f'Distance (in radians)')
ymax = all_lls.max()
#ax.set_ylim(-0.1*ymax,1.1*ymax)
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
#ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.savefig(base_path+'plot_llresampl_frac_full_'+f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}'+'.pdf')


# %%
# teleplot

fig,ax = plt.subplots()
pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
ax.plot(denslist, one_counttele2, label = f't2m', color = 'black')# +- {np.round(pts.std(),1)}')

ax.plot(denslist, [np.mean(all_geo_counttele2[idens]) for idens in range(len(denslist))], label = f'Geomodel 2', color = 'tab:blue')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_geo_counttele2[idens],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_geo_counttele2[idens],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)

ax.plot(denslist, all_iaaft_counttele2.mean(axis = 1), label = f'IAAFT', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_iaaft_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_iaaft_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)

ax.plot(denslist, all_boot_counttele2.mean(axis = 1), label = f'bootstrap', color = 'tab:green')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_boot_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_boot_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)

ax.legend()
ax.set_ylabel(f'Fraction of links in bundles')
ax.set_xlabel('Density')

plt.savefig(base_path + f'resampling_1tm_countteleplot_full_{np.round(pts.mean(),1)}_' +f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.pdf')

# %%
# joint plot for paper
fig,axs = plt.subplots(1,2, figsize=(2*onefigsize[0],onefigsize[1]))
axs[0].plot(llbins[1:-1], one_lls[lineidx,1:], label = 't2m',color='black')
axs[0].plot(one_llquant1[lineidx,:].mean(),0,'x',color = axs[0].lines[-1].get_color())
axs[0].plot(one_llquant2[lineidx,:].mean(),0,'o',color = axs[0].lines[-1].get_color())


axs[0].plot(llbins[1:-1], all_geo_lls[lineidx,1:,:].mean(axis = 1), label = 'Geomodel 2', color = 'tab:blue')
axs[0].fill_between(llbins[1:-1], all_geo_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_geo_lls[lineidx,1:,:].std(axis = 1), all_geo_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_geo_lls[lineidx,1:,:].std(axis = 1),color = axs[0].lines[-1].get_color(), alpha = 0.4)
axs[0].plot(np.mean(all_geo_llquant1[lineidx]),0,'x',color = axs[0].lines[-1].get_color())
axs[0].plot(np.mean(all_geo_llquant2[lineidx]),0,'o',color = axs[0].lines[-1].get_color())

axs[0].plot(llbins[1:-1], all_iaaft_lls[lineidx,1:,:].mean(axis = 1), label = 'IAAFT', color = 'tab:orange')
axs[0].fill_between(llbins[1:-1], all_iaaft_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_iaaft_lls[lineidx,1:,:].std(axis = 1), all_iaaft_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_iaaft_lls[lineidx,1:,:].std(axis = 1),color = axs[0].lines[-1].get_color(), alpha = 0.4)
axs[0].plot(all_iaaft_llquant1[lineidx,:].mean(),0,'x',color = axs[0].lines[-1].get_color())
axs[0].plot(all_iaaft_llquant2[lineidx,:].mean(),0,'o',color = axs[0].lines[-1].get_color())

axs[0].plot(llbins[1:-1], all_boot_lls[lineidx,1:,:].mean(axis = 1), label = 'Bootstrap',color='tab:green')
axs[0].fill_between(llbins[1:-1], all_boot_lls[lineidx,1:,:].mean(axis = 1) - 2 *all_boot_lls[lineidx,1:,:].std(axis = 1), all_boot_lls[lineidx,1:,:].mean(axis = 1) + 2 *all_boot_lls[lineidx,1:,:].std(axis = 1),color = axs[0].lines[-1].get_color(), alpha = 0.4)
axs[0].plot(all_boot_llquant1[lineidx,:].mean(),0,'x',color = axs[0].lines[-1].get_color())
axs[0].plot(all_boot_llquant2[lineidx,:].mean(),0,'o',color = axs[0].lines[-1].get_color())
#upper_ll = llbins[np.where(true_lls[lineidx,:]>0)[0][-1]]
#axs[0].plot((upper_ll,upper_ll), (0,1000000), linestyle = '--', color = axs[0].lines[-1].get_color())
axs[0].legend()
axs[0].set_ylabel(f'Fraction of links')
axs[0].set_xlabel(f'Distance (in radians)')
ymax = all_lls.max()
#axs[0].set_ylim(-0.1*ymax,1.1*ymax)
axs[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
#axs[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
axs[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
axs[1].plot(denslist, one_counttele2, label = f't2m', color = 'black')# +- {np.round(pts.std(),1)}')

axs[1].plot(denslist, [np.mean(all_geo_counttele2[idens]) for idens in range(len(denslist))], label = f'Geomodel 2', color = 'tab:blue')# +- {np.round(pts.std(),1)}')
axs[1].fill_between(denslist, [quantile(all_geo_counttele2[idens],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_geo_counttele2[idens],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)

axs[1].plot(denslist, all_iaaft_counttele2.mean(axis = 1), label = f'IAAFT', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
axs[1].fill_between(denslist, [quantile(all_iaaft_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_iaaft_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)

axs[1].plot(denslist, all_boot_counttele2.mean(axis = 1), label = f'bootstrap', color = 'tab:green')# +- {np.round(pts.std(),1)}')
axs[1].fill_between(denslist, [quantile(all_boot_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_boot_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)

axs[1].legend()
axs[1].set_ylabel(f'Fraction of links in bundles')
axs[1].set_xlabel('Density')
axs = enumerate_subplots(axs,fontsize = 16)
plt.savefig(base_path + f'joint_resamplingplot_{np.round(pts.mean(),1)}_' +f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.pdf')


# %%
# teleplot
fig,ax = plt.subplots()
pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
ax.plot(denslist, one_countrobusttele2, label = f't2m', color = 'black')# +- {np.round(pts.std(),1)}')

ax.plot(denslist, [np.mean(all_geo_countrobusttele2[idens]) for idens in range(len(denslist))], label = f'Geomodel 2', color = 'tab:blue')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_geo_countrobusttele2[idens],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_geo_countrobusttele2[idens],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)

ax.plot(denslist, all_iaaft_countrobusttele2.mean(axis = 1), label = f'IAAFT', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_iaaft_countrobusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_iaaft_countrobusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)

ax.plot(denslist, all_boot_countrobusttele2.mean(axis = 1), label = f'bootstrap', color = 'tab:green')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_boot_countrobusttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_boot_countrobusttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)

ax.legend()
ax.set_ylabel(f'Fraction of links in bundles')
ax.set_xlabel('Density')

plt.savefig(base_path + f'resampling_robusttele_countteleplot_full_{np.round(pts.mean(),1)}_' +f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.pdf')

# %%
# teleplot
fig,ax = plt.subplots()
pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
ax.plot(denslist, one_countrobusttele2raw, label = f't2m', color = 'black')# +- {np.round(pts.std(),1)}')

ax.plot(denslist, [np.mean(all_geo_countrobusttele2raw[idens]) for idens in range(len(denslist))], label = f'Geomodel 2', color = 'tab:blue')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_geo_countrobusttele2raw[idens],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_geo_countrobusttele2raw[idens],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)


ax.plot(denslist, all_iaaft_countrobusttele2raw.mean(axis = 1), label = f'IAAFT', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_iaaft_countrobusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_iaaft_countrobusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)

ax.plot(denslist, all_boot_countrobusttele2raw.mean(axis = 1), label = f'bootstrap', color = 'tab:green')# +- {np.round(pts.std(),1)}')
ax.fill_between(denslist, [quantile(all_boot_countrobusttele2raw[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_boot_countrobusttele2raw[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)

ax.legend()
ax.set_ylabel(f'Fraction of links in bundles')
ax.set_xlabel('Density')

plt.savefig(base_path + f'resampling_robustteleraw_countteleplot_full_{np.round(pts.mean(),1)}_' +f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.pdf')





# %%
# fig,ax = plt.subplots()
# pts = np.sum(dists <= 2 * dist_equator, axis = 1) #all_counttele2, all_countrobusttele2, all_countrobusttele2raw
# ax.plot(denslist, one_counttele2, label = f't2m', color = 'black')# +- {np.round(pts.std(),1)}')

# ax.plot(denslist, all_iaaft_counttele2.mean(axis = 1), label = f'IAAFT', color = 'tab:orange')# +- {np.round(pts.std(),1)}')
# ax.fill_between(denslist, [quantile(all_iaaft_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_iaaft_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)

# ax.plot(denslist, all_boot_counttele2.mean(axis = 1), label = f'bootstrap', color = 'tab:green')# +- {np.round(pts.std(),1)}')
# ax.fill_between(denslist, [quantile(all_boot_counttele2[idens,:],alpha=0.025) for idens in range(len(denslist))] , [quantile(all_boot_counttele2[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)

# ax.legend()
# ax.set_ylabel(f'Fraction of links in bundles')
# ax.set_xlabel('Density')

# plt.savefig(base_path + f'resampling_1tm_countteleplot_{np.round(pts.mean(),1)}_' +f'{var_name}_{corr_method}_{typ}_w{weighted}_{grid_type}{n_lat}_tol{robust_tolerance}.pdf')

# %%
geoadj = myload(base_path+'geomodeladj_dens0.005_geps0.05_rew10_nlat72_run0.txt')
(geoadj !=0).sum(),all_geo_lls[1,:,0].sum(), one_lls[3,:].sum()