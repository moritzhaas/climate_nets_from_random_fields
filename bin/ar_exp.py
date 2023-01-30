# %%
from matplotlib import rcParams
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
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
#plt.style.use('bmh')

def autocorr(X, t = 1):
    '''
    Computes autocorr at lag 1 between columns of X.
    '''
    d = X.shape[1]
    return np.corrcoef(X[:-t,:], X[t:,:],rowvar = False)[:d,d:]

# def eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx, n_pre = 100):
#     dim = cov.shape[0]
#     data = np.zeros((n_time+n_pre,dim))
#     eigA = np.zeros_like(eigvecs)
#     eigA[pc_idx,pc_idx] = ar_pc
#     A = eigvecs @ eigA @ eigvecs.T
#     eigvals_eps = eigvals
#     eigvals_eps[pc_idx] = eigvals[pc_idx] * (1-ar_pc ** 2)
#     sigma = eigvecs @ np.diag(eigvals_eps) @ eigvecs.T
#     eps = np.random.multivariate_normal(np.zeros(dim), sigma, size=n_time+n_pre)
#     data[0,:] = np.random.multivariate_normal(np.zeros(dim), eigvecs @ np.diag(eigvals) @ eigvecs.T)
#     for i in range(n_pre+n_time-1):
#         data[i+1,:] = A @ data[i,:] + eps[i+1,:]
#     return data[n_pre:,:]

num_runs = 30
longtime = 10000

# Network parameters
n_lat = 18 * 4 # makes resolution of 180 / n_lat degrees
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = False
ranks = False
corr_method='spearman' # 'spearman', 'MI', 'HSIC', 'ES'
pcar = False
#denslist = [0.001,0.01,0.05,0.1,0.2]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.075,0.1,0.125,0.15]
ks = [6,15,30,45, 60,90,120,150,180,210,240,270, 300,450, 600,750,900]
#ks = [5,  10, 65, 125, 250]
if len(denslist) != len(ks) and typ == 'threshold':
    raise RuntimeError('Denslist needs to have same length as ks.')

robust_tolerance = 0.5
# data parameters
distrib = 'igrf'
n_time = 500
nu = 1.5
len_scale = 0.2


exec(open("grid_helper.py").read())

# %%
print('Computing covariance matrix.')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

if pcar:
# diagonal VAR1 coeff of length num_points
    pc_idx = 1
    ar_pc = 0.9
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals, eigvecs = eigvals[np.argsort(eigvals)], eigvecs[:,np.argsort(eigvals)]
    #eigvalssig, eigvecssig = np.linalg.eig(sigma)
    #eigvalssig, eigvecssig = eigvalssig[np.argsort(eigvalssig)], eigvecssig[:,np.argsort(eigvalssig)]
    #np.where(np.abs(eigvalssig-eigvals)>1e-1) # other eigvals are affected although they shouldn't
else:
    ac = 0.2
    ar = ac
    ac2 = 0.7
    ar_coeff = ac * np.ones(num_points)

    if find(f'arcoeffidcs_{num_points}.txt',base_path) == []:
        rd_idcs = np.random.permutation(np.arange(num_points))[:num_points // 2]
        mysave(base_path,f'arcoeffidcs_{num_points}.txt',rd_idcs)
    else:
        rd_idcs = myload(base_path + f'arcoeffidcs_{num_points}.txt')
    ar_coeff[rd_idcs] = ac2 * np.ones(len(rd_idcs))


#datacov = true_g_var(ar_coeff,covcov_dia) pretty wrong?

# %%
arbins = np.linspace(-1,1,21)
ardegs = np.zeros((len(denslist),num_runs,len(arbins)-1,3))
longars = np.zeros((3,len(lat)))
arbiasquotient = np.zeros((len(denslist),num_runs))
arbiasl = np.zeros((len(denslist),num_runs))
arbiash = np.zeros((len(denslist),num_runs))
# %%
# mysave(base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',data)
name = f'{corr_method}_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1.txt'
# def thisname and replace filename
# if pcar:
#     thiisname = name + f'_{typ}_{grid_type}_pcidx{pc_idx}_ar{ar_pc}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}'
# else:
#     thiisname = name + f'_{typ}_{grid_type}_ar{ac}_ar2{ac2}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}'
if os.path.exists(base_path+ f'longcorr_{name}'):
    longcorr = myload(base_path+ f'longcorr_{name}')
else:
    for i in range(len(longars)):
        if pcar:
            longdata = eig_var_process(eigvecs, eigvals, longtime, ar_pc, pc_idx)
        else:
            longdata = diag_var_process(ar_coeff, cov, longtime)
        for j in range(num_points):
            longdata[:,j] -= longdata[:,j].mean()
            longdata[:,j] /= longdata[:,j].std()
        longars[i,:] = np.diag(autocorr(longdata))

    longcorr = compute_empcorr(longdata)
    mysave(base_path,f'longcorr_{name}',longcorr)
# %%
#datacov = true_cov_diag_var(ar_coeff,cov)
# %%
#np.linalg.norm(longcorr-cov, ord = 2),np.linalg.norm(longcorr-datacov, ord = 2),np.linalg.norm(cov-datacov, ord = 2)
# %%
filename = base_path + name[:-4] +'.nc'
curr_time = time.time()

all_densk,all_denskw = np.zeros_like(all_dens),np.zeros_like(all_dens)
all_degsk,all_degskw = np.zeros_like(all_degs),np.zeros_like(all_degs)
ardegsk,ardegskw = np.zeros_like(ardegs),np.zeros_like(ardegs)
karbiash, kwarbiash = np.zeros_like(arbiash),np.zeros_like(arbiash)
karbiasl, kwarbiasl = np.zeros_like(arbiasl),np.zeros_like(arbiasl)
# %%
if os.path.exists(base_path+ f'arstats_long_{name}'):
    all_dens,all_densk,all_denskw, ardegs,ardegsk,ardegskw,arbiash,arbiasl,karbiash,karbiasl, kwarbiash,kwarbiasl,all_degs,all_degsk,all_degskw = myload(base_path+ f'arstats_{name}')
else:
    for run in range(num_runs):
        print(f'Starting {run}th run. ', time.time()-curr_time)
        curr_time = time.time()
        if pcar:
            data = eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx)
        else:
            data = diag_var_process(ar_coeff, cov, n_time)
        for j in range(num_points):
            data[:,j] -= data[:,j].mean()
            data[:,j] /= data[:,j].std()
        
        print('Computing similarity matrix. ', time.time()-curr_time)
        curr_time = time.time()
        emp_corr = compute_empcorr(data, similarity=corr_method)
        
        for i in range(len(denslist)):
            density = denslist[i]
            k = ks[i]
            adj = get_adj(emp_corr, density, weighted=weighted)
            kadj = knn_adj(emp_corr, k, weighted = False)
            kwadj = knn_adj(emp_corr, k, weighted= True)

            deg = adj.sum(axis = 0)/(adj.shape[0]-1)
            kdeg = kadj.sum(axis = 0)/(kadj.shape[0]-1)
            kwdeg = kwadj.sum(axis = 0)/(kwadj.shape[0]-1)
            all_dens[i,run] = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
            all_densk[i,run] = kadj.sum()/ (kadj.shape[0]*(kadj.shape[0]-1))    
            all_denskw[i,run] = kwadj.sum()/ (kwadj.shape[0]*(kwadj.shape[0]-1))    

            #print('Avg. norm. deg of higher AR vs avg. norm. deg', deg[rd_idcs].mean(), deg.mean())
            #ar_bias[i,run] = deg[rd_idcs].mean() / deg.mean()
            ars = np.diag(autocorr(data))
            for iar in range(len(arbins)-1):
                thesedegs = deg[np.logical_and(arbins[iar]< longars.mean(axis=0), longars.mean(axis=0) <= arbins[iar+1])]
                thesedegsk = kdeg[np.logical_and(arbins[iar]< longars.mean(axis=0), longars.mean(axis=0) <= arbins[iar+1])]
                thesedegskw = kwdeg[np.logical_and(arbins[iar]< longars.mean(axis=0), longars.mean(axis=0) <= arbins[iar+1])]
                ardegs[i, run, iar, :] = thesedegs.mean(), thesedegs.std(), len(thesedegs)
                ardegsk[i, run, iar, :] = thesedegsk.mean(), thesedegsk.std(), len(thesedegsk)
                ardegskw[i, run, iar, :] = thesedegskw.mean(), thesedegskw.std(), len(thesedegsk)

            #print('Computing graphstats. ', time.time()-curr_time)
            if not pcar:
                arbiash[i,run] = deg[rd_idcs].mean() 
                arbiasl[i,run] = deg[ar_coeff == ac].mean()
                karbiash[i,run] = kdeg[rd_idcs].mean() 
                karbiasl[i,run] = kdeg[ar_coeff == ac].mean()
                kwarbiash[i,run] = kwdeg[rd_idcs].mean() 
                kwarbiasl[i,run] = kwdeg[ar_coeff == ac].mean()
            curr_time = time.time()
            # plot degree distribution, ll distribution, density, density given l, furthest eps stable teleconnection
            degnum = np.histogram(deg, bins=degbins)[0]
            kdegnum = np.histogram(kdeg, bins=degbins)[0]
            kwdegnum = np.histogram(kwdeg, bins=degbins)[0]
            all_degs[i,:,run] = degnum/num_points
            all_degsk[i,:,run] = kdegnum/num_points
            all_degskw[i,:,run] = kwdegnum/num_points

    ar_stats = [all_dens,all_densk,all_denskw, ardegs,ardegsk,ardegskw,arbiash,arbiasl,karbiash,karbiasl, kwarbiash,kwarbiasl,all_degs,all_degsk,all_degskw]
    mysave(base_path, f'arstats_long_{name}',ar_stats)
# %%
kwarbiash *= (all_densk.mean(axis=1) / all_denskw.mean(axis=1)).reshape((17,-1))
kwarbiasl *= (all_densk.mean(axis=1) / all_denskw.mean(axis=1)).reshape((17,-1))

# %%
import seaborn as sns
from climnet.myutils import *

def quantile(arr, alpha = 0.95):
    return np.sort(arr)[int(np.ceil(alpha*len(arr)-1))]
ar_coeff = ac * np.ones(num_points)

if find(f'arcoeffidcs_{num_points}.txt',base_path) == []:
    rd_idcs = np.random.permutation(np.arange(num_points))[:num_points // 2]
    mysave(base_path,f'arcoeffidcs_{num_points}.txt',rd_idcs)
else:
    rd_idcs = myload(base_path + f'arcoeffidcs_{num_points}.txt')
ar_coeff[rd_idcs] = ac2 * np.ones(len(rd_idcs))
rd_idcs = np.sort(rd_idcs)
lowidcs = np.where(ar_coeff < 0.5)[0]# np.where(longars.mean(axis = 0) < 0.5)[0]

if pcar:
    data = eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx)
else:
    data = diag_var_process(ar_coeff, cov, n_time)
for j in range(num_points):
    data[:,j] -= data[:,j].mean()
    data[:,j] /= data[:,j].std()

emp_corr = compute_empcorr(data, similarity=corr_method)
corrhigh = upper(cov[rd_idcs,:][:,rd_idcs])
corrlow = upper(cov[lowidcs,:][:,lowidcs])
emphigh = upper(emp_corr[rd_idcs,:][:,rd_idcs])
emplow = upper(emp_corr[lowidcs,:][:,lowidcs])
corrbins = np.linspace(0,0.8,17)
highquant = np.zeros(len(corrbins)-1)
lowquant = np.zeros(len(corrbins)-1)
highmedian = np.zeros(len(corrbins)-1)
lowmedian = np.zeros(len(corrbins)-1)
highdown = np.zeros(len(corrbins)-1)
lowdown = np.zeros(len(corrbins)-1)
for i in range(len(corrbins)-1):
    thesecorrhigh = emphigh[np.where(np.logical_and(corrhigh > corrbins[i], corrhigh < corrbins[i+1]))[0]]
    thesecorrlow = emplow[np.where(np.logical_and(corrlow > corrbins[i], corrlow < corrbins[i+1]))[0]]
    if len(thesecorrhigh) == 0:
        highquant[i] =np.nan
        highmedian[i] = np.nan
        highdown[i]= np.nan
    else:
        highquant[i] = quantile(thesecorrhigh, alpha = 0.95)
        highmedian[i] = quantile(thesecorrhigh, alpha = 0.5)
        highdown[i] = quantile(thesecorrhigh, alpha = 0.05)
        print(len(thesecorrhigh), thesecorrhigh.min(),thesecorrhigh.max(),thesecorrhigh.mean(),quantile(thesecorrhigh, alpha = 0.5),quantile(thesecorrhigh, alpha = 0.95))
    if len(thesecorrlow) == 0:
        lowquant[i] = np.nan
        lowmedian[i] = np.nan
        lowdown[i] = np.nan
    else:
        lowquant[i] = quantile(thesecorrlow,alpha=0.95)
        lowmedian[i] = quantile(thesecorrlow,alpha=0.5)
        lowdown[i] = quantile(thesecorrlow, alpha = 0.05)
        print(len(thesecorrlow), thesecorrlow.min(),thesecorrlow.max(), thesecorrlow.mean(),quantile(thesecorrlow, alpha = 0.5),quantile(thesecorrlow, alpha = 0.95))


# %%
# plot jointly
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)


sparseadj = get_adj(emp_corr, 0.01)
sparse_thres = sparseadj[sparseadj != 0].min()
denseadj = get_adj(emp_corr,0.1)
dense_thres = denseadj[denseadj != 0].min()
corrbins = np.linspace(0,0.8,17)
fig,axs = plt.subplots(1,3,figsize=(3*onefigsize[0],1.2*onefigsize[1]))
axs[0].plot(binstoplot(corrbins), highmedian, label='high AR')
axs[0].fill_between(binstoplot(corrbins),highdown, highquant,alpha=0.4)
axs[0].plot(binstoplot(corrbins), lowmedian, label = 'low AR')#, color = 'red')
axs[0].fill_between(binstoplot(corrbins),lowdown, lowquant,alpha=0.4)#, color = 'red')
axs[0].plot((binstoplot(corrbins)[0],0.77), (dense_thres,dense_thres),'--',color = 'black')
axs[0].plot((binstoplot(corrbins)[0],0.77), (sparse_thres,sparse_thres),'--',color = 'black')
axs[0].text(0.55, dense_thres-0.1, 'dens=0.1')
axs[0].text(0.52, sparse_thres-0.1, 'dens=0.01')
axs[0].set_xlabel('True correlation')
axs[0].set_ylabel(f'Emp. Spearman corr.')
axs[0].legend()

ids = np.random.permutation(np.arange(len(upper(emp_corr[rd_idcs,:][:,rd_idcs]))))#[:2500]#[:2500000]
levels = np.array([0.001,0.01, 0.05, 0.1, 0.5])

#sns.kdeplot(x = corrhigh[ids], y=upper(emp_corr[rd_idcs,:][:,rd_idcs])[ids], levels=levels, color="tab:blue", linewidths=1, label = 'high AR')
#sns.kdeplot(x = corrlow[ids], y=upper(emp_corr[lowidcs,:][:,lowidcs])[ids], levels=levels, color="tab:orange", linewidths=1, label = 'low AR')
sns.kdeplot(ax=axs[1],x = upper(longcorr[rd_idcs,:][:,rd_idcs])[ids], y=upper(emp_corr[rd_idcs,:][:,rd_idcs])[ids], levels=levels, color="tab:blue", linewidths=1, label = 'high AR')
sns.kdeplot(ax=axs[1],x = upper(longcorr[lowidcs,:][:,lowidcs])[ids], y=upper(emp_corr[lowidcs,:][:,lowidcs])[ids], levels=levels, color="tab:orange", linewidths=1, label = 'low AR')
axs[1].legend(loc=(0.07,0.705))
axs[1].set_ylim(0,0.8)
axs[1].set_xlim(0,0.8)
axs[1].set_xlabel('True correlation')
axs[1].set_ylabel(f'Emp. Spearman corr')
#plt.plot(binstoplot(corrbins), highquant, color = adjust_lightness('tab:blue', amount=0.6))
#axs[0,1].plot(binstoplot(corrbins), lowquant, color = adjust_lightness('tab:orange', amount=0.6))
#axs[0,1].text(0.8, highquant[-2], '95% quantile', color = 'b')
#axs[0,1].text(0.8, lowquant[-2]-0.03, '95% quantile', color = 'red')
axs[1].plot((0,1), (dense_thres,dense_thres),'--', color = 'black')
axs[1].plot((0,1), (sparse_thres,sparse_thres),'--',color = 'black')
axs[1].text(0.575, dense_thres-0.08, 'dens=0.1')
axs[1].text(0.55, sparse_thres-0.08, 'dens=0.01')


all_dens,all_densk,all_denskw, ardegs,ardegsk,ardegskw,arbiash,arbiasl,karbiash,karbiasl, kwarbiash,kwarbiasl,all_degs,all_degsk,all_degskw = myload(base_path+ f'arstats_long_{name}')
kwarbiash *= (all_densk.mean(axis=1) / all_denskw.mean(axis=1)).reshape((17,-1))
kwarbiasl *= (all_densk.mean(axis=1) / all_denskw.mean(axis=1)).reshape((17,-1))

# axs[0,1].rcParams.update({'font.size': 20})
# axs[0,1].errorbar(all_dens.mean(axis = 1), arbiasquotient.mean(axis = 1), yerr = 2 * arbiasquotient.std(axis = 1))
# axs[0,1].xlabel('Density')
# axs[0,1].ylabel('mean(deg(highAR) / deg(lowAR))')
# axs[0,1].title('AR bias quotient for degree')
# axs[0,1].savefig(base_path + f'arbiasquotient_{num_runs}runs_{n_time}time_0.2_0.7.pdf',dpi=150)

# delta_ar = (arbiash-arbiasl)/(arbiash+arbiasl)
# kdelta_ar = (karbiash-karbiasl)/(karbiash+karbiasl)
# kwdelta_ar = (kwarbiash-kwarbiasl)/(kwarbiash+kwarbiasl)

# axs[2].plot(denslist, delta_ar.mean(axis=1), label='unw. threshold', color = 'tab:blue')
# axs[2].fill_between(denslist,[quantile(delta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(delta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)
# axs[2].plot(all_densk.mean(axis=1), kdelta_ar.mean(axis=1), label='unw. kNN', color = 'tab:orange')
# axs[2].fill_between(all_densk.mean(axis=1),[quantile(kdelta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(kdelta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
# axs[2].plot(all_densk.mean(axis=1), kwdelta_ar.mean(axis=1), label='weighted kNN', color = 'tab:green')
# axs[2].fill_between(all_densk.mean(axis=1),[quantile(kwdelta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(kwdelta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
# axs[2].set_xlabel('Density')
# axs[2].set_ylabel('Avg. norm. degree')
# axs[2].set_xlim(-0.005,0.2)
# axs[2].legend(loc='upper right')

p1,=axs[2].plot(denslist, arbiash.mean(axis=1)/denslist, label='high AR, unw. threshold', color = adjust_lightness('tab:blue', amount=1.5))
axs[2].fill_between(denslist, [quantile(arbiash[idens,:]/denslist[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(arbiash[idens,:]/denslist[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
p2,=axs[2].plot(denslist, arbiasl.mean(axis=1)/denslist, label='low AR, unw. threshold', color = adjust_lightness('tab:blue', amount=0.5))
axs[2].fill_between(denslist, [quantile(arbiasl[idens,:]/denslist[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(arbiasl[idens,:]/denslist[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
p3,=axs[2].plot(all_densk.mean(axis=1), karbiash.mean(axis=1)/all_densk.mean(axis=1), label='high AR, unw. kNN', color = adjust_lightness('tab:orange', amount=1.5))
axs[2].fill_between(all_densk.mean(axis=1), [quantile(karbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(karbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
p4,=axs[2].plot(all_densk.mean(axis=1), karbiasl.mean(axis=1)/all_densk.mean(axis=1), label='low AR, unw. kNN', color = adjust_lightness('tab:orange', amount=0.5))
axs[2].fill_between(all_densk.mean(axis=1), [quantile(karbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(karbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
p5,=axs[2].plot(all_densk.mean(axis=1), kwarbiash.mean(axis=1)/all_densk.mean(axis=1), label='high AR, weighted kNN', color = adjust_lightness('tab:green', amount=1.5))
axs[2].fill_between(all_densk.mean(axis=1), [quantile(kwarbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(kwarbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
p6,=axs[2].plot(all_densk.mean(axis=1), kwarbiasl.mean(axis=1)/all_densk.mean(axis=1), label='low AR, weighted kNN', color = adjust_lightness('tab:green', amount=0.5))
axs[2].fill_between(all_densk.mean(axis=1), [quantile(kwarbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(kwarbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = axs[2].lines[-1].get_color(), alpha = 0.4)
axs[2].set_xlabel('Density')
axs[2].set_ylabel('Avg. norm. degree')
axs[2].set_xlim(-0.005,0.2)
l = axs[2].legend([(p1, p2),(p3,p4),(p5,p6)], ['Unw. threshold','Unw. kNN','Weighted kNN'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
#axs[2].legend()
axs = enumerate_subplots(axs,fontsize=16)
plt.savefig(base_path + f'joint_arbias_newtry_{name}.pdf')




# %%
fig2,ax = plt.subplots()
p1,=ax.plot(denslist, arbiash.mean(axis=1)/denslist, label='high AR, unw. threshold', color = adjust_lightness('tab:blue', amount=1.5))
ax.fill_between(denslist, [quantile(arbiash[idens,:]/denslist[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(arbiash[idens,:]/denslist[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
p2,=ax.plot(denslist, arbiasl.mean(axis=1)/denslist, label='low AR, unw. threshold', color = adjust_lightness('tab:blue', amount=0.5))
ax.fill_between(denslist, [quantile(arbiasl[idens,:]/denslist[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(arbiasl[idens,:]/denslist[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
p3,=ax.plot(all_densk.mean(axis=1), karbiash.mean(axis=1)/all_densk.mean(axis=1), label='high AR, unw. kNN', color = adjust_lightness('tab:orange', amount=1.5))
ax.fill_between(all_densk.mean(axis=1), [quantile(karbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(karbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
p4,=ax.plot(all_densk.mean(axis=1), karbiasl.mean(axis=1)/all_densk.mean(axis=1), label='low AR, unw. kNN', color = adjust_lightness('tab:orange', amount=0.5))
ax.fill_between(all_densk.mean(axis=1), [quantile(karbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(karbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
p5,=ax.plot(all_densk.mean(axis=1), kwarbiash.mean(axis=1)/all_densk.mean(axis=1), label='high AR, weighted kNN', color = adjust_lightness('tab:green', amount=1.5))
ax.fill_between(all_densk.mean(axis=1), [quantile(kwarbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(kwarbiash[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
p6,=ax.plot(all_densk.mean(axis=1), kwarbiasl.mean(axis=1)/all_densk.mean(axis=1), label='low AR, weighted kNN', color = adjust_lightness('tab:green', amount=0.5))
ax.fill_between(all_densk.mean(axis=1), [quantile(kwarbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.025) for idens in range(len(denslist))], [quantile(kwarbiasl[idens,:]/all_densk.mean(axis=1)[idens],alpha=0.975) for idens in range(len(denslist))],color = ax.lines[-1].get_color(), alpha = 0.4)
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
ax.set_xlim(-0.005,0.2)
l = ax.legend([(p1, p2),(p3,p4),(p5,p6)], ['Unw. threshold','Unw. kNN','Weighted kNN'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)

# %%
# does not output suff quality
from PIL import Image

foo = Image.open(base_path+'jointlinkplots_seed1657551699_matern_nu1.5_len0.2_ar0_fekete36_time1000_var1.png') 
foo.size
# %%
# downsize the image with an ANTIALIAS filter (gives the highest quality)
foo = foo.resize((4387,903),Image.ANTIALIAS)

#foo.save('path/to/save/image_scaled.jpg', quality=95)  # The saved downsized image size is 24.8kb
foo.save(base_path+'jointlinkplots_seed1657551699_matern_nu1.5_len0.2_ar0_fekete36_time1000_var1_optimized.png', optimize=True, quality=95)


# %%
plotdegs = np.zeros((len(arbins)-1,len(denslist),2))
for i in range(len(arbins)-1):
    for idens in range(len(denslist)):
        for run in range(num_runs):
            if ardegs[idens,run,i,2] > 0:
                plotdegs[i,idens,:] += ardegs[idens,run,i,:2] * ardegs[idens,run,i,2]
        if ardegs[idens,run,i,2].sum()>0:
            plotdegs[i,idens,:] /= ardegs[idens, :, i, 2].sum()
        else:
            plotdegs[i,idens,:] = np.nan
for idens in range(len(denslist)):
    plotdegs[:,idens,0] /= np.nanmean(plotdegs[:,idens,0])
    plotdegs[:,idens,1] /= np.nanmean(plotdegs[:,idens,1])


for i in range(len(denslist)):
    plt.plot(binstoplot(arbins), plotdegs[:,i,0], label = f'dens={all_dens[i,:].mean()}')
    plt.fill_between(binstoplot(arbins), plotdegs[:,i,0]-2*plotdegs[:,i,1], plotdegs[:,i,0]+2*plotdegs[:,i,1], alpha = 0.4)

plt.legend()
plt.xlabel('True autocorrelation')
plt.ylabel('Degree')
#plt.xlim(0,1)
plt.ylim(-0.1,2)

# %%
# plt.rcParams.update({'font.size': 20})
# plt.errorbar(all_dens.mean(axis = 1), arbiasquotient.mean(axis = 1), yerr = 2 * arbiasquotient.std(axis = 1))
# plt.xlabel('Density')
# plt.ylabel('mean(deg(highAR) / deg(lowAR))')
# plt.title('AR bias quotient for degree')
# plt.savefig(base_path + f'arbiasquotient_{num_runs}runs_{n_time}time_0.2_0.7.pdf',dpi=150)
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)
fig,ax = plt.subplots()
p1,=ax.plot(denslist, arbiash.mean(axis=1), label='high AR, unw. threshold', color = adjust_lightness('tab:blue', amount=1.5))
ax.fill_between(denslist, arbiash.mean(axis=1)-2*arbiash.std(axis=1), arbiash.mean(axis=1)+2*arbiash.std(axis=1),color = ax.lines[-1].get_color(), alpha = 0.4)
p2,=ax.plot(denslist, arbiasl.mean(axis=1), label='low AR, unw. threshold', color = adjust_lightness('tab:blue', amount=0.5))
ax.fill_between(denslist, arbiasl.mean(axis=1)-2*arbiasl.std(axis=1), arbiasl.mean(axis=1)+2*arbiasl.std(axis=1),color = ax.lines[-1].get_color(), alpha = 0.4)
p3,=ax.plot(all_densk.mean(axis=1), karbiash.mean(axis=1), label='high AR, unw. kNN', color = adjust_lightness('tab:orange', amount=1.5))
ax.fill_between(all_densk.mean(axis=1), karbiash.mean(axis=1)-2*karbiash.std(axis=1), karbiash.mean(axis=1)+2*karbiash.std(axis=1),color = ax.lines[-1].get_color(), alpha = 0.4)
p4,=ax.plot(all_densk.mean(axis=1), karbiasl.mean(axis=1), label='low AR, unw. kNN', color = adjust_lightness('tab:orange', amount=0.5))
ax.fill_between(all_densk.mean(axis=1), karbiasl.mean(axis=1)-2*karbiasl.std(axis=1), karbiasl.mean(axis=1)+2*karbiasl.std(axis=1), color = ax.lines[-1].get_color(),alpha = 0.4)
p5,=ax.plot(all_densk.mean(axis=1), kwarbiash.mean(axis=1), label='high AR, weighted kNN', color = adjust_lightness('tab:green', amount=1.5))
ax.fill_between(all_densk.mean(axis=1), kwarbiash.mean(axis=1)-2*kwarbiash.std(axis=1), kwarbiash.mean(axis=1)+2*kwarbiash.std(axis=1),color = ax.lines[-1].get_color(), alpha = 0.4)
p6,=ax.plot(all_densk.mean(axis=1), kwarbiasl.mean(axis=1), label='low AR, weighted kNN', color = adjust_lightness('tab:green', amount=0.5))
ax.fill_between(all_densk.mean(axis=1), kwarbiasl.mean(axis=1)-2*kwarbiasl.std(axis=1), kwarbiasl.mean(axis=1)+2*kwarbiasl.std(axis=1),color = ax.lines[-1].get_color(), alpha = 0.4)
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
ax.set_xlim(-0.005,0.2)
l = ax.legend([(p1, p2),(p3,p4),(p5,p6)], ['Unw. threshold','Unw. kNN','Weighted kNN'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
#ax.legend()
#plt.savefig(base_path + f'arbias_normalized_{name}.pdf')

# %%
# arbiash[i,run] = deg[rd_idcs].mean() 
# arbiasl[i,run] = deg[ar_coeff == ac].mean()
# delta_ar is difference between avg degrees divided by total avg. which is possible because same amount of high and low
delta_ar = 2* (arbiash-arbiasl)/(arbiash+arbiasl)
kdelta_ar = 2* (karbiash-karbiasl)/(karbiash+karbiasl)
kwdelta_ar = 2* (kwarbiash-kwarbiasl)/(kwarbiash+kwarbiasl)


from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)
fig,ax = plt.subplots()
ax.plot(denslist, delta_ar.mean(axis=1), label='unw. threshold', color = 'tab:blue')
ax.fill_between(denslist,[quantile(delta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(delta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:blue', alpha = 0.4)
ax.plot(all_densk.mean(axis=1), kdelta_ar.mean(axis=1), label='unw. kNN', color = 'tab:orange')
ax.fill_between(all_densk.mean(axis=1),[quantile(kdelta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(kdelta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:orange', alpha = 0.4)
ax.plot(all_densk.mean(axis=1), kwdelta_ar.mean(axis=1), label='weighted kNN', color = 'tab:green')
ax.fill_between(all_densk.mean(axis=1),[quantile(kwdelta_ar[idens,:],alpha=0.025) for idens in range(len(denslist))], [quantile(kwdelta_ar[idens,:],alpha=0.975) for idens in range(len(denslist))], color = 'tab:green', alpha = 0.4)
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
ax.set_xlim(-0.005,0.2)
plt.legend(loc='upper right')

# %%
plt.plot(denslist,arbiash.mean(axis=1)/arbiasl.mean(axis=1))
plt.plot(all_densk.mean(axis=1),karbiash.mean(axis=1)/karbiasl.mean(axis=1))
plt.plot(all_densk.mean(axis=1),kwarbiash.mean(axis=1)/kwarbiasl.mean(axis=1))
# %%
ks,all_densk.mean(axis=1)
# %%
longars.mean(axis = 0), longars.std(axis = 0)

# %%
run =0
print(f'Starting {run}th run. ', time.time()-curr_time)
curr_time = time.time()
if pcar:
    data = eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx)
else:
    data = diag_var_process(ar_coeff, cov, n_time)
for j in range(num_points):
    data[:,j] -= data[:,j].mean()
    data[:,j] /= data[:,j].std()

print('Computing similarity matrix. ', time.time()-curr_time)
curr_time = time.time()
emp_corr = compute_empcorr(data, similarity=corr_method)
low_idcs = [i for i in np.arange(num_points) if i not in rd_idcs]
acorr = np.diag(autocorr(data))
print(acorr[rd_idcs].max(), acorr[rd_idcs].min(), acorr[longars.mean(axis=0) < 0.5].max(), acorr[low_idcs].max())
acorr[low_idcs].mean(),acorr[low_idcs].std(), acorr[rd_idcs].mean(),acorr[rd_idcs].std()

#datacorr = np.diag(1/np.sqrt(np.diag(datacov))) @ datacov @ np.diag(1/np.sqrt(np.diag(datacov)))
# %%
# is the distribution within low autocorr and within high autocorr the same?
#Asphharm = np.array([(i+1) ** (-gamma) for i in range(K+1)])
rd_idcs = np.sort(rd_idcs)
lowidcs = np.where(longars.mean(axis = 0) < 0.5)[0]
corr = cov#np.corrcoef(longdata.T)#datacorr
corrhigh = upper(corr[rd_idcs,:][:,rd_idcs])
corrlow = upper(corr[lowidcs,:][:,lowidcs])

llbins = np.linspace(0,np.pi,51)
highdists = upper(dists[rd_idcs,:][:,rd_idcs])
lowdists = upper(dists[lowidcs,:][:,lowidcs])

truecorrs = kernel(np.concatenate((np.zeros((1,3)),np.concatenate((binstoplot(llbins).reshape((-1,1)), np.zeros((len(llbins)-1,2)) ),axis=1 ))))[0,1:]
highmean, highstd = np.zeros(50), np.zeros(50)
lowmean, lowstd = np.zeros(50), np.zeros(50)
for i in range(len(llbins)-1):
    thesehighcorr = corrhigh[np.where(np.logical_and(highdists > llbins[i], highdists < llbins[i+1]))[0]]
    theselowcorr = corrlow[np.where(np.logical_and(lowdists > llbins[i], lowdists < llbins[i+1]))[0]]
    highmean[i] = thesehighcorr.mean()
    highstd[i] = thesehighcorr.std()
    lowmean[i] = theselowcorr.mean()
    lowstd[i] = theselowcorr.std()
    
plt.plot(binstoplot(llbins), highmean, label = 'highcorr')
plt.fill_between(binstoplot(llbins), highmean-2*highstd, highmean+2*highstd, alpha = 0.4)
plt.plot(binstoplot(llbins), lowmean, label = 'lowcorr')
plt.fill_between(binstoplot(llbins), lowmean-2*lowstd, lowmean+2*lowstd, alpha = 0.4)
plt.plot(binstoplot(llbins), truecorrs, label = f'truecorrs', linestyle = '--')
plt.legend()
plt.xlabel('Distance (in radians)')
plt.ylabel('Correlation')
plt.savefig(base_path+ f'corr_vals_anisoar_{name}.pdf')

# %%
ids = np.arange(500)
plt.scatter(corrhigh[ids], upper(emp_corr[rd_idcs,:][:,rd_idcs])[ids], label = 'high')

plt.scatter(corrlow[ids], upper(emp_corr[lowidcs,:][:,lowidcs])[ids], label = 'low')
plt.legend()
plt.xlabel('True correlation')
plt.ylabel(f'Empirical {corr_method} correlation')


sparseadj = get_adj(emp_corr, 0.01)
sparse_thres = sparseadj[sparseadj != 0].min()
denseadj = get_adj(emp_corr,0.1)
dense_thres = denseadj[denseadj != 0].min()
# %%
# plt.plot(binstoplot(corrbins), highmedian, label='high AR')
# plt.fill_between(binstoplot(corrbins),highdown, highquant,alpha=0.4)
# plt.plot(binstoplot(corrbins), lowmedian, label = 'low AR')#, color = 'red')
# plt.fill_between(binstoplot(corrbins),lowdown, lowquant,alpha=0.4)#, color = 'red')
# plt.plot((binstoplot(corrbins)[0],0.77), (dense_thres,dense_thres),'--',color = 'black')
# plt.plot((binstoplot(corrbins)[0],0.77), (sparse_thres,sparse_thres),'--',color = 'black')
# plt.text(0.55, dense_thres-0.1, 'dens=0.1')
# plt.text(0.52, sparse_thres-0.1, 'dens=0.01')
# plt.xlabel('True correlation')
# plt.ylabel(f'Emp. Spearman corr.')
# plt.legend()
#plt.savefig(base_path + f'true_vs_emp{corr_method}_quantiles_{name}.pdf')



# %%
ids = np.random.permutation(np.arange(len(corrhigh)))#[:2500]#[:2500000]
levels = np.array([0.001,0.01, 0.05, 0.1, 0.5])
f, ax = plt.subplots()
sns.kdeplot(x = corrhigh[ids], y=upper(emp_corr[rd_idcs,:][:,rd_idcs])[ids], levels=levels, color="tab:blue", linewidths=1, label = 'high AR')
sns.kdeplot(x = corrlow[ids], y=upper(emp_corr[lowidcs,:][:,lowidcs])[ids], levels=levels, color="tab:orange", linewidths=1, label = 'low AR')
plt.legend(loc=(0.07,0.705))
plt.ylim(0,0.8)
plt.xlim(0,0.8)
plt.xlabel('True correlation')
plt.ylabel(f'Emp. Spearman corr')
#plt.plot(binstoplot(corrbins), highquant, color = adjust_lightness('tab:blue', amount=0.6))
#plt.plot(binstoplot(corrbins), lowquant, color = adjust_lightness('tab:orange', amount=0.6))
#plt.text(0.8, highquant[-2], '95% quantile', color = 'b')
#plt.text(0.8, lowquant[-2]-0.03, '95% quantile', color = 'red')
plt.plot((0,1), (dense_thres,dense_thres),'--', color = 'black')
plt.plot((0,1), (sparse_thres,sparse_thres),'--',color = 'black')
plt.text(0.575, dense_thres-0.08, 'dens=0.1')
plt.text(0.55, sparse_thres-0.08, 'dens=0.01')
#plt.savefig(base_path + f'truecorrvsemp{corr_method}_nu{nu}_{len_scale}_{n_time}time_{ac}_{ac2}.pdf',dpi=150)



# %%
len(corrhigh)
# %%
noiseidx = 1
deg = sparseadj.sum(axis = 1)
degquants = np.where(np.sort(deg) == deg[noiseidx])[0]
degquantile = (degquants[0] + degquants[-1]) / (2 * len(deg))
degquantile

# %%
longars[longars>0.5].mean(),longars[longars>0.5].std(),longars[longars<0.5].mean(),longars[longars<0.5].std(),

# %%
# denslist = [0.001,0.01,0.05,0.1,0.2]
# filter_string = '*matern_nu0.5_len0.1_ar0_fekete72_time100_*'
# similarity = 'spearman'
# betwbins2 = np.linspace(0,0.17, num_bins + 1)

# max_runs = 2 #len(find(filter_string, base_path+ 'empdata/'))
# if max_runs != 30:
#     print('max_runs not 30 but ', max_runs)
# all_betw2 = np.zeros((len(denslist), num_bins, max_runs))
# density2 = np.zeros((len(denslist),max_runs))

# for irun,nam in enumerate(find(filter_string, base_path+ 'empdata/')):
#     #load several runs and plot true vs emp as well as ar bias
#     ...









# #%%
# def get_near_psd(A, epsilon = 1e-8):
#     C = (A + A.T)/2
#     eigval, eigvec = np.linalg.eig(C)
#     eigval[eigval < 0] = epsilon
#     return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

# def _getAplus(A):
#     eigval, eigvec = np.linalg.eig(A)
#     Q = np.matrix(eigvec)
#     xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
#     return Q*xdiag*Q.T

# def _getPs(A, W=None):
#     W05 = np.matrix(W**.5)
#     return  W05.I * _getAplus(W05 * A * W05) * W05.I

# def _getPu(A, W=None):
#     Aret = np.array(A.copy())
#     Aret[W > 0] = np.array(W)[W > 0]
#     return np.matrix(Aret)

# def nearPD(A, nit=10):
#     n = A.shape[0]
#     W = np.identity(n) 
# # W is the matrix used for the norm (assumed to be Identity matrix here)
# # the algorithm should work for any diagonal W
#     deltaS = 0
#     Yk = A.copy()
#     for k in range(nit):
#         Rk = Yk - deltaS
#         Xk = _getPs(Rk, W=W)
#         deltaS = Xk - Rk
#         Yk = _getPu(Xk, W=W)
#     return Yk

# def near_psd(x, epsilon=1e-8):
#     '''
#     Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

#     Parameters
#     ----------
#     x : array_like
#       Covariance/correlation matrix
#     epsilon : float
#       Eigenvalue limit (usually set to zero to ensure positive definiteness)

#     Returns
#     -------
#     near_cov : array_like
#       closest positive definite covariance/correlation matrix

#     Notes
#     -----
#     Document source
#     http://www.quarchome.org/correlationmatrix.pdf

#     '''

#     if min(np.linalg.eigvals(x)) > epsilon:
#         return x

#     # Removing scaling factor of covariance matrix
#     n = x.shape[0]
#     var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
#     y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

#     # getting the nearest correlation matrix
#     eigval, eigvec = np.linalg.eig(y)
#     val = np.matrix(np.maximum(eigval, epsilon))
#     vec = np.matrix(eigvec)
#     T = 1/(np.multiply(vec, vec) * val.T)
#     T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
#     B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
#     near_corr = B*B.T    

#     # returning the scaling factors
#     near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
#     return near_cov

# A = cov - np.diag(ar_coeff) @ cov @ np.diag(ar_coeff)
# A1,A2,A3 = get_near_psd(A), near_psd(A), nearPD(A)
# def get_near_psd(A, epsilon = 1e-8):
#     C = (A + A.T)/2
#     eigval, eigvec = np.linalg.eig(C)
#     eigval[eigval < 0] = epsilon
#     return eigvec.dot(np.diag(eigval)).dot(eigvec.T)
# A1 = get_near_psd(cov - np.diag(ar_coeff) @ cov @ np.diag(ar_coeff))
# actual_cov = np.zeros_like(A1)
# for i in range(30):
#     rec = np.diag(ar_coeff**i)
#     actual_cov += rec @ A1 @ rec

# print(np.linalg.norm(actual_cov - cov, 2), np.linalg.norm(actual_cov - cov) )
# plt.hist(longars.mean(axis=0), bins = 100)
# # %%
# '''dataiid = np.zeros((longtime,1483))
# rvs = np.random.normal(size = (len(A) ** 2, longtime + 1))
# T = isotropic_grf(A, rvs = rvs)
# for j in range(len(lat)):
#     la = lat[j]
#     lo = lon[j]
#     dataiid[:,j] = T(la,lo)[1:]
#     dataiid[:,j] -= data[:,j].mean()
#     dataiid[:,j] /= data[:,j].std()
# '''

# W2_normal(0,0, cov, actual_cov), W2_normal(0,0, cov, np.eye(cov.shape[0]))

#np.linalg.norm(A-A1,2), np.linalg.norm(A-A2,2), np.linalg.norm(A-A3,2)
# %%
perm_idcs = np.random.permutation(num_points)
curr_time = time.time()
perm_maxavgdeg = maxavgdeg(true_adj[perm_idcs,:][:,perm_idcs],nbhds2)
print(time.time()-curr_time)
# %%
