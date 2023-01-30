# %%
import os, fnmatch, pickle
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
#from multiprocessing import Pool
from collections import Counter
from sklearn.gaussian_process.kernels import Matern


#irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
start_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
corr_method='spearman'
weighted = False
ranks = False

ar = 0
ar2 = None
var = 10
robust_tolerance = 0.2
n_perm = 10

num_runs = 30
n_time = 500
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5 # irrelevant
len_scale = 0.1 # irrelevant

denslist = [0.001,0.01,0.05,0.1,0.2]#,0.2]
ks = [6, 60, 300, 600,1200]#,1200]


#filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

# %%
#grid_helper and calc_true
exec(open("grid_helper.py").read())
#exec(open("calc_true.py").read())

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

# comppose telestats
all_tele1 = np.zeros((numth, num_runs))
all_tele2 = np.zeros((numth, num_runs))
#all_tele3 = np.zeros((numth, num_runs))
all_robusttele2 = np.zeros((numth, num_runs))
all_robusttele2raw = np.zeros((numth, num_runs))
#all_robusttele3 = np.zeros((numth, num_runs))
# all_llquant1 = np.zeros((numth, num_runs))
# all_llquant2 = np.zeros((numth, num_runs))
all_mad = np.zeros((numth, num_runs))
all_shufflemad = np.zeros((numth,n_perm,num_runs))
all_dens = np.zeros((numth, num_runs))


#%%
hyperlist = []
for filename in find(f'telestats_part0_*time{n_time}*', base_path+ 'bundlestats/',nosub=True):
    hyperparams = filename.split('telestats_part0_',1)[1]
    if hyperparams not in hyperlist:
        hyperlist.append(hyperparams)


#numth = 5
for hyperparams in hyperlist:
    these_files = list(set(find('*'+hyperparams, base_path+ 'bundlestats/')) & set(find('telestats_part*', base_path+ 'bundlestats/',nosub=True)))
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))
    try:
        one_dens,one_tele1,one_tele2, one_robusttele2,one_robusttele2raw, one_mad, one_shufflemad = myload(these_files[0])
    except:
        print('Too few variables.',these_files[0])
        continue
    numth = one_dens.shape[0]
    all_tele1 = np.zeros((numth, these_runs))
    all_tele2 = np.zeros((numth, these_runs))
    #all_tele3 = np.zeros((numth, these_runs))
    all_robusttele2 = np.zeros((numth, these_runs))
    all_robusttele2raw = np.zeros((numth, these_runs))
    #all_robusttele3 = np.zeros((numth, these_runs))
    # all_llquant1 = np.zeros((numth, these_runs))
    # all_llquant2 = np.zeros((numth, these_runs))
    all_mad = np.zeros((numth, these_runs))
    all_shufflemad = np.zeros((numth,n_perm,these_runs))
    all_dens = np.zeros((numth, these_runs))
    for i,nam in enumerate(these_files):
        one_dens,one_tele1,one_tele2, one_robusttele2,one_robusttele2raw, one_mad, one_shufflemad = myload(nam)
        all_dens[:,i] = one_dens
        all_tele1[:,i] = one_tele1
        all_tele2[:,i] = one_tele2
        all_robusttele2[:,i] = one_robusttele2
        all_robusttele2raw[:,i] = one_robusttele2raw
        all_mad[:,i] = one_mad
        all_shufflemad[:,:,i] = one_shufflemad
    all_stats = [all_dens,all_tele1,all_tele2, all_robusttele2,all_robusttele2raw, all_mad, all_shufflemad] # all_tele3, all_robusttele3,
    outfilename = f'alltelestats_{these_runs}_' + hyperparams
    mysave(base_path+ 'bundlestats/',outfilename,all_stats)
    if not os.path.exists(base_path+'bundlestats/old/'):
        os.mkdir(base_path+'bundlestats/old/')
    for nam in these_files:
        filenam = nam.split('bundlestats/',1)[1]
        os.replace(nam, base_path+'bundlestats/old/'+filenam)

# %%
hyperlist = []
for filename in find('othertelestats_part0_*', base_path+ 'bundlestats/',nosub=True):#, base_path+ 'bundlestats/',nosub=True):
    hyperparams = filename.split('othertelestats_part0_',1)[1]
    if hyperparams not in hyperlist:
        hyperlist.append(hyperparams)
print(len(hyperlist))
#numth = 5
for hyperparams in hyperlist:
    these_files = list(set(find('*'+hyperparams, base_path+ 'bundlestats/')) & set(find('othertelestats_part*', base_path+ 'bundlestats/',nosub=True)))
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    one_fdr,one_tele2,one_counttele2, one_countrobusttele2, one_countrobusttele2raw = myload(these_files[0])
    numth = one_fdr.shape[0]
    all_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = [np.zeros((numth, these_runs)) for _ in range(5)]
    #all_dens = np.zeros((numth, these_runs))
    for i,nam in enumerate(these_files):
        one_fdr,one_tele2,one_counttele2, one_countrobusttele2, one_countrobusttele2raw = myload(nam)
        all_fdr[:,i] = one_fdr
        all_tele2[:,i] = one_tele2
        all_counttele2[:,i] = one_counttele2
        all_countrobusttele2[:,i] = one_countrobusttele2
        all_countrobusttele2raw[:,i] = one_countrobusttele2raw
    all_stats = [all_fdr,all_tele2,all_counttele2, all_countrobusttele2, all_countrobusttele2raw] # all_tele3, all_robusttele3,
    outfilename = f'allothertelestats_{these_runs}_' + hyperparams
    mysave(base_path+ 'bundlestats/',outfilename,all_stats)
    if not os.path.exists(base_path+'bundlestats/old/'):
        os.mkdir(base_path+'bundlestats/old/')
    for nam in these_files:
        filenam = nam.split('bundlestats/',1)[1]
        os.replace(nam, base_path+'bundlestats/old/'+filenam)


# %%
hyperlist = []
for filename in find('uniformtelestats_part0_*', base_path+ 'bundlestats/',nosub=True):
    hyperparams = filename.split('uniformtelestats_part0_',1)[1]
    if hyperparams not in hyperlist:
        hyperlist.append(hyperparams)
print(len(hyperlist))
#numth = 5
for hyperparams in hyperlist:
    these_files = list(set(find('*'+hyperparams, base_path+ 'bundlestats/')) & set(find('uniformtelestats_part*', base_path+ 'bundlestats/',nosub=True)))
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    one_dens,one_counttele2, one_countrobusttele2, one_countrobusttele2raw = myload(these_files[0])
    numth = one_dens.shape[0]
    all_dens,all_counttele2, all_countrobusttele2, all_countrobusttele2raw = [np.zeros((numth, these_runs)) for _ in range(4)]
    #all_dens = np.zeros((numth, these_runs))
    for i,nam in enumerate(these_files):
        one_dens,one_counttele2, one_countrobusttele2, one_countrobusttele2raw = myload(nam)
        all_dens[:,i] = one_dens
        all_counttele2[:,i] = one_counttele2
        all_countrobusttele2[:,i] = one_countrobusttele2
        all_countrobusttele2raw[:,i] = one_countrobusttele2raw
    all_stats = [all_dens,all_counttele2, all_countrobusttele2, all_countrobusttele2raw] # all_tele3, all_robusttele3,
    outfilename = f'alluniformtelestats_{these_runs}_' + hyperparams
    mysave(base_path+ 'bundlestats/',outfilename,all_stats)
    if not os.path.exists(base_path+'bundlestats/old/'):
        os.mkdir(base_path+'bundlestats/old/')
    for nam in these_files:
        filenam = nam.split('bundlestats/',1)[1]
        os.replace(nam, base_path+'bundlestats/old/'+filenam)


# %%
import fnmatch
# compose graphstats
gslist = []
for filename in find('graphstats_part0_*', base_path+ 'graphstats/',nosub=True):
    hyperparams = filename.split('graphstats_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('*'+hyperparams, base_path+ 'graphstats/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*allgraphstats*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    one_dens,one_densw, one_degs, one_lls, one_ccs,one_degws, one_llws,one_ccws, one_spls, plink,one_llquant1,one_llquant2,one_llwquant1,one_llwquant2 = myload(find('*'+hyperparams, base_path+'graphstats/')[0])
    numth = one_dens.shape[0]
    all_degs, all_lls,all_degws, all_llws, all_ccs,all_ccws, all_spls = [np.zeros((numth, num_bins,  these_runs)) for _ in range(7)]
    all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = [np.zeros((numth,  these_runs)) for _ in range(4)]
    all_spls = np.zeros((numth, len(splbins)-1,  these_runs))
    all_dens = np.zeros((numth,  these_runs))
    all_densw = np.zeros((numth,  these_runs))
    total_plink = np.zeros((numth, n_dresol))
    
    for i,nam in enumerate(these_files):
        one_dens,one_densw, one_degs, one_lls, one_ccs,one_degws, one_llws,one_ccws, one_spls, plink,one_llquant1,one_llquant2,one_llwquant1,one_llwquant2 = myload(nam)
        all_dens[:,i] = one_dens
        all_densw[:,i] = one_densw
        all_degs[:,:,i] = one_degs
        all_degws[:,:,i] = one_degws
        all_lls[:,:,i] = one_lls
        all_llws[:,:,i] = one_llws
        all_ccs[:,:,i] = one_ccs
        all_ccws[:,:,i] = one_ccws
        all_spls[:,:,i] = one_spls
        all_llquant1[:,i] = one_llquant1
        all_llwquant1[:,i] = one_llwquant1
        all_llquant2[:,i] = one_llquant2
        all_llwquant2[:,i] = one_llwquant2
        total_plink += plink

    for ibin in range(n_dresol):
        if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
            total_plink[:,ibin] /= (these_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())
    outfilename = f'graphstats/allgraphstats_{these_runs}_' + hyperparams   
    all_stats = [all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, total_plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2]
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_stats, fp)
    if not os.path.exists(base_path+'graphstats/old/'):
        os.mkdir(base_path+'graphstats/old/')
    for nam in these_files:
        filenam = nam.split('graphstats/',1)[1]
        os.replace(nam, base_path+'graphstats/old/'+filenam)

# %%
import fnmatch
# compose graphstats
gslist = []
for filename in find('graphstats_part0_*noise*', base_path+ 'graphstats/',nosub=True):
    hyperparams = filename.split('graphstats_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('*'+hyperparams, base_path+ 'graphstats/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*allgraphstats*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    noise = np.float64(hyperparams.split('noise',1)[1].split('_',1)[0])
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    cov /= (1+noise ** 2)
    one_dens,one_densw, one_degs, one_lls, one_ccs,one_degws, one_llws,one_ccws, one_spls, plink,one_llquant1,one_llquant2,one_llwquant1,one_llwquant2 = myload(find('*'+hyperparams, base_path+'graphstats/')[0])
    numth = one_dens.shape[0]
    all_degs, all_lls,all_degws, all_llws, all_ccs,all_ccws, all_spls = [np.zeros((numth, num_bins,  these_runs)) for _ in range(7)]
    all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = [np.zeros((numth,  these_runs)) for _ in range(4)]
    all_spls = np.zeros((numth, len(splbins)-1,  these_runs))
    all_dens = np.zeros((numth,  these_runs))
    all_densw = np.zeros((numth,  these_runs))
    total_plink = np.zeros((numth, n_dresol))
    
    for i,nam in enumerate(these_files):
        one_dens,one_densw, one_degs, one_lls, one_ccs,one_degws, one_llws,one_ccws, one_spls, plink,one_llquant1,one_llquant2,one_llwquant1,one_llwquant2 = myload(nam)
        all_dens[:,i] = one_dens
        all_densw[:,i] = one_densw
        all_degs[:,:,i] = one_degs
        all_degws[:,:,i] = one_degws
        all_lls[:,:,i] = one_lls
        all_llws[:,:,i] = one_llws
        all_ccs[:,:,i] = one_ccs
        all_ccws[:,:,i] = one_ccws
        all_spls[:,:,i] = one_spls
        all_llquant1[:,i] = one_llquant1
        all_llwquant1[:,i] = one_llwquant1
        all_llquant2[:,i] = one_llquant2
        all_llwquant2[:,i] = one_llwquant2
        total_plink += plink

    for ibin in range(n_dresol):
        if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
            total_plink[:,ibin] /= (these_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())
    outfilename = f'graphstats/allgraphstats_{these_runs}_' + hyperparams   
    all_stats = [all_dens,all_densw, all_degs, all_lls, all_ccs,all_degws, all_llws,all_ccws, all_spls, total_plink,all_llquant1,all_llquant2,all_llwquant1,all_llwquant2]
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_stats, fp)
    if not os.path.exists(base_path+'graphstats/old/'):
        os.mkdir(base_path+'graphstats/old/')
    for nam in these_files:
        filenam = nam.split('graphstats/',1)[1]
        os.replace(nam, base_path+'graphstats/old/'+filenam)

# %%
gslist = []
for filename in find('plink_part0_*', base_path+ 'plink/',nosub=True):
    hyperparams = filename.split('plink_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('plink_part*'+hyperparams, base_path+ 'plink/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*allplink*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    plink = myload(these_files[0])
    numth, n_dresol = plink.shape
    total_plink = np.zeros((numth, n_dresol))
    
    for i,nam in enumerate(these_files):
        plink = myload(nam)
        total_plink += plink

    for ibin in range(n_dresol):
        if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
            total_plink[:,ibin] /= (these_runs *  np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum())
    outfilename = f'plink/allplink_{these_runs}_' + hyperparams   
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(total_plink, fp)
    if not os.path.exists(base_path+'plink/old/'):
        os.mkdir(base_path+'plink/old/')
    for nam in these_files:
        filenam = nam.split('plink/',1)[1]
        os.replace(nam, base_path+'plink/old/'+filenam)


# %%
import fnmatch
# compose fdr
gslist = []
for filename in find('fdr_part0_*', base_path+ 'bundlestats/',nosub=True):
    hyperparams = filename.split('fdr_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('fdr*'+hyperparams, base_path+ 'bundlestats/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*all*')]
    #these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*part*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    one_fdr = np.array(myload(these_files[0]))
    #print(one_fdr)
    numth = one_fdr.shape[0]
    all_fdr = np.zeros((numth,len(these_files)))
    for i,nam in enumerate(these_files):
        one_fdr = myload(nam)
        try:
            all_fdr[:,i] = one_fdr
        except:
            print(i, one_fdr)

    outfilename = f'bundlestats/allfdr_{these_runs}_' + hyperparams   
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_fdr, fp)
    if not os.path.exists(base_path+'bundlestats/old/'):
        os.mkdir(base_path+'bundlestats/old/')
    for nam in these_files:
        filenam = nam.split('bundlestats/',1)[1]
        os.replace(nam, base_path+'bundlestats/old/'+filenam)

# %%
import fnmatch
# compose fdr
gslist = []
for filename in find('fdr_part0_*', base_path+ 'fdr/',nosub=True):
    hyperparams = filename.split('fdr_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('fdr_part*'+hyperparams, base_path+ 'fdr/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*all*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    one_fdr = np.array(myload(these_files[0]))
    #print(one_fdr)
    numth = one_fdr.shape[0]
    all_fdr = np.zeros((numth,len(these_files)))
    stop = 0
    for i,nam in enumerate(these_files):
        one_fdr = myload(nam)
        try:
            all_fdr[:,i] = one_fdr
        except:
            print(i, one_fdr)
            stop = 1
    if stop == 0:
        outfilename = f'fdr/allfdr_{these_runs}_' + hyperparams   
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump(all_fdr, fp)
        if not os.path.exists(base_path+'fdr/old/'):
            os.mkdir(base_path+'fdr/old/')
        for nam in these_files:
            filenam = nam.split('fdr/',1)[1]
            os.replace(nam, base_path+'fdr/old/'+filenam)




# %%
import tqdm
ac = 0.2
ac2 = 0.7
n_lat = 4 * 18
num_points = gridstep_to_numpoints(180/n_lat)
quants,ar_coeff,alphas = myload(find('arbiasquants_part0_*',base_path+'signif/')[0])
gslist = []
for filename in find('arbiasquants_part0_*', base_path+ 'signif/',nosub=True):
    hyperparams = filename.split('arbiasquants_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('arbiasquants_part*'+hyperparams, base_path+ 'signif/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*all*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
        continue
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))
        continue
    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
    
    # orig_data = myload(find(f'arbiasdata_nu{nu}_len{len_scale}_ar{ac}_ac2{ac2}_{grid_type}{n_lat}_time{n_time}*',base_path+'empdata/')[0])
    # data = 1 * orig_data
    # # save with nam in name
    # for j in range(len(lat)):
    #     data[:,j] -= orig_data[:,j].mean()
    #     data[:,j] /= orig_data[:,j].std()
    # emp_corr = compute_empcorr(data,'pearson')
    quants,data,alphas = myload(these_files[0])
    all_quantadj = [np.zeros((num_points,num_points)) for _ in range(len(alphas))]
    stop = 0
    for i,nam in tqdm.tqdm(enumerate(these_files)):
        quants,data,alphas = myload(nam)
        if data.shape == (n_time,num_points):
            emp_corr = compute_empcorr(data)
        else:
            stop = 1
            print(nam,data.shape, 'arcoeff instead of data')
            break
        keys = list(quants.keys())
        for ialpha in range(len(alphas)):
            for i in range(len(keys)):
                all_quantadj[ialpha][keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] > quants[keys[i]][ialpha])
            all_quantadj[ialpha]=np.maximum(all_quantadj[ialpha].T,all_quantadj[ialpha])
    outfilename = f'signif/allarbiasquants_{these_runs}_' + hyperparams   
    if stop == 0:
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump([all_quantadj,data,alphas], fp)
        if not os.path.exists(base_path+'signif/old/'):
            os.mkdir(base_path+'signif/old/')
        for nam in these_files:
            filenam = nam.split('signif/',1)[1]
            os.replace(nam, base_path+'signif/old/'+filenam)

# %%
import tqdm
ac = 0.2
ac2 = 0.7
n_time=100
n_lat = 4 * 18
num_points = gridstep_to_numpoints(180/n_lat)
#quants,ar_coeff,alphas = myload(find('arbiasquants_iaaft_part0_*',base_path+'signif/')[0])
gslist = []
for filename in find('arbiasquants_iaaft_part0_*', base_path+ 'signif/',nosub=True):
    hyperparams = filename.split('arbiasquants_iaaft_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('arbiasquants_iaaft_part*'+hyperparams, base_path+ 'signif/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*all*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
        continue
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))
        continue
    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
    
    # orig_data = myload(find(f'arbiasdata_nu{nu}_len{len_scale}_ar{ac}_ac2{ac2}_{grid_type}{n_lat}_time{n_time}*',base_path+'empdata/')[0])
    # data = 1 * orig_data
    # # save with nam in name
    # for j in range(len(lat)):
    #     data[:,j] -= orig_data[:,j].mean()
    #     data[:,j] /= orig_data[:,j].std()
    # emp_corr = compute_empcorr(data,'pearson')
    quants, mean_std ,data, alphas = myload(these_files[0])
    all_quantadj = [np.zeros((num_points,num_points)) for _ in range(len(alphas))]
    all_zscores = -9999 * np.ones((num_points,num_points))
    stop = 0
    for i,nam in tqdm.tqdm(enumerate(these_files)):
        quants, mean_std ,data, alphas = myload(nam)
        if data.shape == (n_time,num_points):
            emp_corr = compute_empcorr(data)
        else:
            stop = 1
            print(nam,data.shape, 'arcoeff instead of data')
            break
        keys = list(quants.keys())
        for i in range(len(keys)):
            all_zscores[keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] - mean_std[keys[i]][0])/mean_std[keys[i]][1]
        
        for ialpha in range(len(alphas)):
            for i in range(len(keys)):
                all_quantadj[ialpha][keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] > quants[keys[i]][ialpha])
            all_quantadj[ialpha]=np.maximum(all_quantadj[ialpha].T,all_quantadj[ialpha])
    outfilename = f'signif/alliaaftzscores_{these_runs}_' + hyperparams 
    with open(base_path + outfilename, "wb") as fp:
        pickle.dump([all_zscores,data,alphas], fp)
    outfilename = f'signif/alliaaft_arbiasquants_{these_runs}_' + hyperparams   
    if stop == 0:
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump([all_quantadj,data,alphas], fp)
        if not os.path.exists(base_path+'signif/old/'):
            os.mkdir(base_path+'signif/old/')
        for nam in these_files:
            filenam = nam.split('signif/',1)[1]
            os.replace(nam, base_path+'signif/old/'+filenam)


gslist = []
for filename in find('arbiasquants_shuffles_part0_*', base_path+ 'signif/',nosub=True):
    hyperparams = filename.split('arbiasquants_shuffles_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('arbiasquants_shuffles_part*'+hyperparams, base_path+ 'signif/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*all*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
        continue
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))
        continue
    nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
        cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    else:
        cov = kernel(spherical2cartesian(lon,lat))
        mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
    
    # orig_data = myload(find(f'arbiasdata_nu{nu}_len{len_scale}_ar{ac}_ac2{ac2}_{grid_type}{n_lat}_time{n_time}*',base_path+'empdata/')[0])
    # data = 1 * orig_data
    # # save with nam in name
    # for j in range(len(lat)):
    #     data[:,j] -= orig_data[:,j].mean()
    #     data[:,j] /= orig_data[:,j].std()
    # emp_corr = compute_empcorr(data,'pearson')
    quants, mean_std ,data, alphas = myload(these_files[0])
    all_quantadj = [np.zeros((num_points,num_points)) for _ in range(len(alphas))]
    all_zscores = -9999 * np.ones((num_points,num_points))
    stop = 0
    for i,nam in tqdm.tqdm(enumerate(these_files)):
        quants, mean_std ,data, alphas = myload(nam)
        if data.shape == (n_time,num_points):
            emp_corr = compute_empcorr(data)
        else:
            stop = 1
            print(nam,data.shape, 'arcoeff instead of data')
            break
        keys = list(quants.keys())
        for i in range(len(keys)):
            all_zscores[keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] - mean_std[keys[i]][0])/mean_std[keys[i]][1]
        
        for ialpha in range(len(alphas)):
            for i in range(len(keys)):
                all_quantadj[ialpha][keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] > quants[keys[i]][ialpha])
            all_quantadj[ialpha]=np.maximum(all_quantadj[ialpha].T,all_quantadj[ialpha])
    outfilename = f'signif/allshuffleszscores_{these_runs}_' + hyperparams 
    with open(base_path + outfilename, "wb") as fp:
        pickle.dump([all_zscores,data,alphas], fp)
    outfilename = f'signif/allshuffles_arbiasquants_{these_runs}_' + hyperparams   
    if stop == 0:
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump([all_quantadj,data,alphas], fp)
        if not os.path.exists(base_path+'signif/old/'):
            os.mkdir(base_path+'signif/old/')
        for nam in these_files:
            filenam = nam.split('signif/',1)[1]
            os.replace(nam, base_path+'signif/old/'+filenam)

# %%
base_path = '../../climnet_output/'
num_runs = 30
gslist = []
for filename in find('rfstats_part0_*', base_path+ 'rfstats/',nosub=True):
    hyperparams = filename.split('rfstats_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('rfstats_part*'+hyperparams, base_path+ 'rfstats/', nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*allrfstats*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))

    #nu = np.float64(hyperparams.split('nu',1)[1].split('_',1)[0])
    #len_scale = hyperparams.split('len',1)[1].split('_',1)[0]
    n_lat = np.int64(hyperparams.split('fekete',1)[1].split('_',1)[0])
    num_points = gridstep_to_numpoints(180/n_lat)
    # kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
    # if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
    #     cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
    # else:
    #     cov = kernel(spherical2cartesian(lon,lat))
    #     mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)

    th_dens,thavgcorrmean,thavgcorrstd,th_lengths1,th_lengths2 = myload(these_files[0])
    print(th_dens.shape,th_lengths1.shape)
    nulen, lenlen, thlen = th_dens.shape
    all_dens,all_lengths1,all_lengths2 = [np.zeros((nulen, lenlen,thlen,len(these_files))) for _ in range(3)]
    #all_lengths1,all_lengths2  = [np.zeros((nulen, lenlen,thlen,num_points,len(these_files))) for _ in range(2)]
    avgcorrmean, avgcorrstd = [np.zeros((nulen,lenlen,len(these_files))) for _ in range(2)]
    # for i,nam in enumerate(these_files):
    #     th_dens,thavgcorrmean,thavgcorrstd,th_lengths1,th_lengths2 = myload(nam)
    #     print(th_lengths1[1,1,2,:].std())
    for i,nam in enumerate(these_files):
        th_dens,thavgcorrmean,thavgcorrstd,th_lengths1,th_lengths2 = myload(nam)
        all_dens[:,:,:,i] = th_dens
        avgcorrmean[:,:,i] = thavgcorrmean
        avgcorrstd[:,:,i] = thavgcorrstd
        all_lengths1[:,:,:,i] = th_lengths1.mean(axis=3)
        all_lengths2[:,:,:,i] = th_lengths2.mean(axis=3)

    outfilename = f'rfstats/allrfstats_{these_runs}_' + hyperparams   
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump([all_dens,avgcorrmean,avgcorrstd,all_lengths1,all_lengths2], fp)
    if not os.path.exists(base_path+'rfstats/old/'):
        os.mkdir(base_path+'rfstats/old/')
    for nam in these_files:
        filenam = nam.split('rfstats/',1)[1]
        os.replace(nam, base_path+'rfstats/old/'+filenam)


# %%
import fnmatch
# compose graphstats
gslist = []
for filename in find('resampling_part0_*tol0.2*', base_path,nosub=True):
    hyperparams = filename.split('resampling_part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)

for hyperparams in gslist:
    print(hyperparams)
    these_files = find('*'+hyperparams, base_path, nosub=True)
    these_files = [f for f in these_files if not fnmatch.fnmatch(f,'*allresampling*')]
    these_runs = len(these_files)
    if these_runs<num_runs:
        print(hyperparams+' has only '+ str(these_runs))
    elif these_runs > num_runs:
        print(hyperparams+' has even '+ str(these_runs))


    num_links,one_lls,one_llquant1, one_llquant2, one_counttele2, one_countrobusttele2, one_countrobusttele2raw,one_iaaft_lls,one_iaaft_llquant1, one_iaaft_llquant2, one_iaaft_counttele2, one_iaaft_countrobusttele2, one_iaaft_countrobusttele2raw,one_boot_lls,one_boot_llquant1, one_boot_llquant2, one_boot_counttele2, one_boot_countrobusttele2, one_boot_countrobusttele2raw = myload(find('resampling*'+hyperparams, base_path)[0])
    numth = one_lls.shape[0]
    all_iaaft_lls,all_boot_lls = [np.zeros((numth, num_bins,  these_runs)) for _ in range(2)]
    all_iaaft_llquant1,all_iaaft_llquant2,all_iaaft_counttele2,all_iaaft_countrobusttele2,all_iaaft_countrobusttele2raw,all_boot_llquant1,all_boot_llquant2,all_boot_counttele2,all_boot_countrobusttele2,all_boot_countrobusttele2raw = [np.zeros((numth,  these_runs)) for _ in range(10)]
    
    for i,nam in enumerate(these_files):
        num_links,one_lls,one_llquant1, one_llquant2, one_counttele2, one_countrobusttele2, one_countrobusttele2raw,one_iaaft_lls,one_iaaft_llquant1, one_iaaft_llquant2, one_iaaft_counttele2, one_iaaft_countrobusttele2, one_iaaft_countrobusttele2raw,one_boot_lls,one_boot_llquant1, one_boot_llquant2, one_boot_counttele2, one_boot_countrobusttele2, one_boot_countrobusttele2raw =  myload(nam)
        all_iaaft_lls[:,:,i] = one_iaaft_lls
        all_iaaft_llquant1[:,i] = one_iaaft_llquant1
        all_iaaft_llquant2[:,i] = one_iaaft_llquant2
        all_iaaft_counttele2[:,i] = one_iaaft_counttele2
        all_iaaft_countrobusttele2[:,i] = one_iaaft_countrobusttele2
        all_iaaft_countrobusttele2raw[:,i] = one_iaaft_countrobusttele2raw
        all_boot_lls[:,:,i] = one_boot_lls
        all_boot_llquant1[:,i] = one_boot_llquant1
        all_boot_llquant2[:,i] = one_boot_llquant2
        all_boot_counttele2[:,i] = one_boot_counttele2
        all_boot_countrobusttele2[:,i] = one_boot_countrobusttele2
        all_boot_countrobusttele2raw[:,i] = one_boot_countrobusttele2raw

    outfilename = f'allresampling_{these_runs}_' + hyperparams   
    all_stats = [num_links,one_lls,one_llquant1, one_llquant2, one_counttele2, one_countrobusttele2, one_countrobusttele2raw,all_iaaft_lls,all_iaaft_llquant1, all_iaaft_llquant2, all_iaaft_counttele2, all_iaaft_countrobusttele2, all_iaaft_countrobusttele2raw,all_boot_lls,all_boot_llquant1, all_boot_llquant2, all_boot_counttele2, all_boot_countrobusttele2, all_boot_countrobusttele2raw]
    with open(base_path + outfilename, "wb") as fp:   #Pickling
        pickle.dump(all_stats, fp)

# %%














# %%
find(f'*_nu{nu}_len{len_scale}_ar{ac}_ac2{ac2}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')

# %%
find('fdr_part*'+hyperparams, base_path+ 'fdr/', nosub=True)
gslist
# %%
# for recovering files from old:

# for nam in find('fdr_part*', base_path+ 'bundlestats/old/',nosub=True):
#     filenam = nam.split('bundlestats/old/',1)[1]
#     os.replace(nam, base_path+'bundlestats/'+filenam)


#%%
find('graphstats_*', base_path+'graphstats/')
# %%
graphstatname=find('graphstats_igrf_*', base_path)[0]
all_dens, all_degs, all_lls, all_ccs,all_ccws, all_spls, plink,all_llquant1,all_llquant2=myload(graphstatname)
# %%
# plt.plot(binstoplot(llbins),all_lls[0,:,0])
# plt.plot(binstoplot(llbins),all_lls[1,:,0])
# plt.plot(binstoplot(llbins),all_lls[2,:,0])
# plt.plot(binstoplot(llbins),all_lls[3,:,0])
# plt.plot(binstoplot(llbins),all_lls[4,:,0])
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(0,0.3)
