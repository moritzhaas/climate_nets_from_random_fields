#%%
import os, fnmatch, pickle
import numpy as np
import matplotlib.pyplot as plt
from climnet.grid import regular_lon_lat, regular_lon_lat_step, FeketeGrid
from climnet.myutils import *
import time
from collections import Counter
from sklearn.gaussian_process.kernels import Matern
from sklearn.covariance import GraphicalLassoCV, MinCovDet
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
#plt.style.use('bmh')
irun = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', irun)
curr_time = time.time()

base_path = '../../climnet_output/'
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18 * 4
typ ='threshold'
corr_method='LWlin'
weighted = False
ranks = False

ar = 0
ar2 = None
var = 1
robust_tolerance = 0.2
n_perm = 10

num_runs = 30
n_time = 100
nus = [0.5,1.5]
len_scales = [0.05,0.1,0.2]
nu = 0.5 # irrelevant
len_scale = 0.1 # irrelevant

denslist = [0.001,0.01,0.05,0.1,0.2]
ks = [6, 60, 300, 600,1200]


filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'

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

# %%
# compute normdistances
corr_methods = ['pearson', 'LWlin', 'LW'] #, 'GL', 'MCD']
frob = np.zeros(len(corr_methods))
l2 = np.zeros(len(corr_methods))

curr_time = time.time()
for nu in nus:
    for len_scale in len_scales:
        print(nu,len_scale,time.time()-curr_time)
        curr_time = time.time()
        from sklearn.gaussian_process.kernels import Matern
        kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
        if find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)!=[]:
            cov = myload(find(f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}*',base_path)[0])
        else:
            cov = kernel(spherical2cartesian(lon,lat))
            mysave(base_path,f'cov_nu{nu}_{len_scale}_{grid_type}{num_points}.txt',cov)
        if corr_method == 'ES':
            empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_*',base_path+'empcorrs/')
            empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time1000_*',base_path+'empdata/')
        else:
            empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empcorrs/')
            empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',base_path+'empdata/')
        
        if irun < len(empdatas):
            orig_data = myload(empdatas[irun])
        else:
            seed = int(time.time())
            np.random.seed(seed)
            orig_data = diag_var_process(ar_coeff, cov, n_time)
            mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',orig_data)
        data = np.sqrt(var) * orig_data
        # save with nam in name
        for j in range(len(lat)):
            data[:,j] -= orig_data[:,j].mean()
            data[:,j] /= orig_data[:,j].std()
        curr_time = time.time()
        lwlinemp_corr = compute_empcorr(data, similarity='LWlin')
        print('LWlin took ', time.time()-curr_time)
        curr_time = time.time()
        lwemp_corr = compute_empcorr(data, similarity='LW')
        print('LW took ', time.time()-curr_time)
        curr_time = time.time()
        pemp_corr = compute_empcorr(data, similarity='pearson')
        print('pearson took ', time.time()-curr_time)
        #curr_time = time.time()
        #GLcov = GraphicalLassoCV().fit(data).covariance_
        #print('GLCV took ', time.time()-curr_time)
        #curr_time = time.time()
        #MCDcov = MinCovDet(random_state=0).fit(data).covariance_
        #print('MCD took ', time.time()-curr_time)
        curr_time = time.time()
        frob[:] = np.linalg.norm(cov - lwlinemp_corr), np.linalg.norm(cov - lwemp_corr), np.linalg.norm(cov - pemp_corr)#, np.linalg.norm(cov - GLcov), np.linalg.norm(cov - MCDcov)
        l2[:] = np.linalg.norm(cov - lwlinemp_corr,ord=2), np.linalg.norm(cov - lwemp_corr,ord=2), np.linalg.norm(cov - pemp_corr,ord=2)#, np.linalg.norm(cov - GLcov,ord=2), np.linalg.norm(cov - MCDcov,ord=2)
        #print(frob.mean(axis = 1), frob.std(axis=1))
        mysave(base_path,f'empmatrixdist_part{irun}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}.txt', [frob,l2])
        for corr_method in corr_methods:
            # if corr_method == 'GL':
            #     emp_corr = GLcov
            # elif corr_method == 'MCD':
            #     emp_corr = MCDcov
            if corr_method == 'LW':
                emp_corr = lwemp_corr
            elif corr_method == 'LWlin':
                emp_corr = lwlinemp_corr
            else:
                emp_corr = pemp_corr
            all_fdr = np.zeros(numth)
            plink = np.zeros((numth, n_dresol))
            for i, density in enumerate(denslist):
                adj = get_adj(emp_corr, density, weighted=False)
                # accumulate over iruns: plink[i,ibin] = (number of links at dens and bin)/(number of possible links at bin)
                distadj = get_adj(5-dists, density,weighted=True)
                maxdist = 5-distadj[distadj!=0].min()
                all_fdr[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
                for ibin in range(n_dresol):
                    if np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1]).sum() > 0:
                        plink[i,ibin] += (adj[np.logical_and(cov >= truecorrbins[ibin], cov < truecorrbins[ibin+1])] != 0).sum()
                    else:
                        plink[i,ibin] = np.nan
            outfilename = f'fdr/fdr_part{irun}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
            with open(base_path + outfilename, "wb") as fp:   #Pickling
                pickle.dump(all_fdr, fp)
            outfilename = f'plink/plink_part{irun}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
            with open(base_path + outfilename, "wb") as fp:   #Pickling
                pickle.dump(plink, fp)
                
                


# %%
gslist = []
for filename in find('empmatrixdist_part0_*', base_path,nosub=True):
    hyperparams = filename.split('part0_',1)[1]
    if hyperparams not in gslist:
        gslist.append(hyperparams)


for hyperparams in gslist:
    print(hyperparams)
    these_files = find('empmatrixdist_part*'+hyperparams, base_path, nosub=True)
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

    if these_files != []:
        one_frob,one_l2 = myload(these_files[0])
    else:
        print(hyperparams, ' no files found.')
        continue
    #print(one_fdr)
    numth = one_frob.shape[0]
    all_frob = np.zeros((numth,len(these_files)))
    all_l2 = np.zeros((numth,len(these_files)))
    stop = 0
    for i,nam in enumerate(these_files):
        one_frob,one_l2 = myload(nam)
        try:
            all_frob[:,i] = one_frob
            all_l2[:,i] = one_l2
        except:
            print(i, one_frob, one_l2)
            stop = 1
    if stop == 0:
        outfilename = f'allempmatrixdist_{these_runs}_' + hyperparams   
        with open(base_path + outfilename, "wb") as fp:   #Pickling
            pickle.dump([all_frob,all_l2], fp)
        # for nam in these_files:
        #     filenam = nam.split('fdr/',1)[1]
        #     os.remove(base_path+'fdr/old/'+filenam)

# %%
find('allempmatrixdist*',base_path)
# %%
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
for corr_method in ['LWlin', 'LW']:
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
    fig, ax = plt.subplots()
    ax.errorbar(idcs-0.13,frobs1,np.array([minfrobs1,maxfrobs1]), fmt='.', capsize=4, color = 'tab:blue', label = 'emp. pearson')
    ax.errorbar(idcs+0.13,frobs2,np.array([minfrobs2,maxfrobs2]), fmt='.', capsize=4, color = 'tab:orange', label = 'LW')
    ax.set_xticks(idcs)
    ax.set_xticklabels([f'nu={thesenus[idx]},\n l={theselen_scales[idx]}' for idx in range(len(thesenames))])
    ax.set_ylabel('Frobenius error')
    ax.legend()
    plt.savefig(base_path + f'plot_frob_{corr_method}_{len(thesenames)}.pdf')
    plt.clf()
    fig, ax = plt.subplots()
    # for idx, distname in enumerate(thesenames):
    #     frob,l2 = myload(distname)
    #     ax.errorbar(idx,l2[-1,:].mean(),np.array([l2[-1,:].min(),l2[-1,:].max()]).reshape((2,1)), fmt='.k', color = 'tab:blue', label = 'emp. pearson')
    #     ax.errorbar(idx, l2[0,:].mean(),np.array([l2[0,:].min(),l2[0,:].max()]).reshape((2,1)), fmt='.k', color = 'tab:orange', label = corr_method)
    ax.errorbar(idcs-0.13,l2s1,np.array([minl2s1,maxl2s1]), fmt='.', capsize=4, color = 'tab:blue', label = 'emp. pearson')
    ax.errorbar(idcs+0.13,l2s2,np.array([minl2s2,maxl2s2]), fmt='.', capsize=4, color = 'tab:orange', label = 'LW')
    ax.set_xticks(idcs)
    ax.set_xticklabels([f'nu={thesenus[idx]},\n l={theselen_scales[idx]}' for idx in range(len(thesenames))])
    ax.set_ylabel('2-norm error')
    ax.legend()
    plt.savefig(base_path + f'plot_l2_{corr_method}_{len(thesenames)}.pdf')
    plt.clf()

# %%
frob,l2 = myload(distname)
frob, l2
# %%
frob[-1,:].mean(),[frob[-1,:].min(),frob[-1,:].max()],frob[0,:].mean(),[frob[0,:].min(),frob[0,:].max()]
# %%
#GLcov = GraphicalLassoCV().fit(data).covariance_
