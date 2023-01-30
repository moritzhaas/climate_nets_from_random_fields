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
#from multiprocessing import Pool
import time
start_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
adjust_fontsize(3)
#plt.style.use('bmh')

curr_time = time.time()

num_runs = 30

# Network parameters
n_lat = 18 * 4 # makes resolution of 180 / n_lat degrees
grid_type = 'fekete' # regular, landsea
typ = 'threshold' # 'knn' 'threshold'
weighted = False
ranks = False
corr_method='pearson' # 'spearman', 'MI', 'HSIC', 'ES'
pcar = False
denslist = np.logspace(-3,np.log(0.25)/np.log(10), num = 30)#denslist = [0.001,0.005,0.01,0.25,0.05,0.1]#np.logspace(-3,np.log(0.25)/np.log(10), num = 20)
# ks = [6, 60, 300, 600,1200]
alphas = [0.5,0.9,0.95,0.99,0.99,0.999]
#ks = [5,  10, 65, 125, 250]
# if len(denslist) != len(ks) and typ == 'threshold':
    # raise RuntimeError('Denslist needs to have same length as ks.')

robust_tolerance = 0.5
# data parameters
distrib = 'igrf'
n_time = 100
nu = 1.5
len_scale = 0.2


exec(open("grid_helper.py").read())

print('Computing covariance matrix.')
from sklearn.gaussian_process.kernels import Matern
kernel = 1.0 * Matern(length_scale=len_scale, nu=nu)
cov = kernel(spherical2cartesian(lon,lat))

ar = 0.2
ac2 = 0.7


longtime = 100000
ar_coeffidcs = myload(base_path + f'arcoeffidcs_{num_points}.txt')
ar_coeff = ar * np.ones(num_points)
ar_coeff[ar_coeffidcs] = ac2

# %%
# subsample data, take mean of correlation estimates and hope for the best?
zscoresshuff,datashuff,alphashuff = myload(find(f'allshuffleszscores_*0.2_0.7*',base_path+'signif/')[0])

sub = 0.6
n_ens = 100

emp_corrsubsample = compute_empcorr(datashuff)
num_points = emp_corrshuff.shape[0]
emp_corrmean = np.zeros_like(emp_corrshuff)

for _ in range(n_ens):
    randtime = np.random.permutation(n_time)[:int(sub*n_time)]
    emp_corrmean += compute_empcorr(datashuff[randtime,:])
emp_corrmean /= n_ens


all_denssubsample,all_fdrsubsample, arbiashsubsample,arbiaslsubsample = [np.zeros(len(denslist)) for _ in range(4)]

if os.path.exists(base_path+'signif/'+f'all_subsamplestats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt'):
    all_denssubsample,all_fdrsubsample,arbiashsubsample,arbiaslsubsample = myload(base_path+'signif/'+f'all_subsamplestats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
else:
    for i, dens in enumerate(denslist):
        adjsubsample = get_adj(emp_corrmean,dens,weighted=False)
        density = adjsubsample.sum()/ (adjsubsample.shape[0]*(adjsubsample.shape[0]-1))
        all_denssubsample[i] = density
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdrsubsample[i] = adjsubsample[np.logical_and(adjsubsample != 0, dists > maxdist)].sum() / adjsubsample[adjsubsample!=0].sum()
        deg = adjsubsample.sum(axis = 0)/(adjsubsample.shape[0]-1)
        arbiashsubsample[i] = deg[ar_coeff != ar].mean() 
        arbiaslsubsample[i] = deg[ar_coeff == ar].mean()
    mysave(base_path+'signif/',f'all_subsamplestats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt',[all_denssubsample,all_fdrsubsample,arbiashsubsample,arbiaslsubsample])



# %%
zscoresshuff,datashuff,alphashuff = myload(find(f'allshuffleszscores_*0.2_0.7*',base_path+'signif/')[0])
emp_corrshuff = compute_empcorr(datashuff)

all_densshuff,all_fdrshuff, all_fdrzshuff,arbiashshuff,arbiaslshuff,arbiashzshuff,arbiaslzshuff = [np.zeros(len(denslist)) for _ in range(7)]

if os.path.exists(base_path+'signif/'+f'all_shuffstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt'):
    all_densshuff,all_fdrshuff,all_fdrzshuff,arbiashshuff,arbiaslshuff,arbiashzshuff,arbiaslzshuff = myload(base_path+'signif/'+f'all_shuffstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
else:
    for i, dens in enumerate(denslist):
        zscoresshuff[np.eye(num_points,dtype=bool)] = 0
        zscoresshuff = np.maximum(zscoresshuff.T,zscoresshuff)
        #print('Starting ', alpha)
        adjzshuff = get_adj(zscoresshuff,dens,weighted=False)#np.maximum(all_quantadj[i],all_quantadj[i].T)
        adjshuff = get_adj(emp_corrshuff,dens,weighted=False)
        density = adjzshuff.sum()/ (adjzshuff.shape[0]*(adjzshuff.shape[0]-1))
        all_densshuff[i] = density
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdrshuff[i] = adjshuff[np.logical_and(adjshuff != 0, dists > maxdist)].sum() / adjshuff[adjshuff!=0].sum()
        all_fdrzshuff[i] = adjzshuff[np.logical_and(adjzshuff != 0, dists > maxdist)].sum() / adjzshuff[adjzshuff!=0].sum()
        deg = adjshuff.sum(axis = 0)/(adjshuff.shape[0]-1)
        degz = adjzshuff.sum(axis=0)/(adjzshuff.shape[0]-1)
        arbiashshuff[i] = deg[ar_coeff != ar].mean() 
        arbiaslshuff[i] = deg[ar_coeff == ar].mean()
        arbiashzshuff[i] = degz[ar_coeff != ar].mean() 
        arbiaslzshuff[i] = degz[ar_coeff == ar].mean()
    mysave(base_path+'signif/',f'all_shuffstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt',[all_densshuff,all_fdrshuff,all_fdrzshuff,arbiashshuff,arbiaslshuff,arbiashzshuff,arbiaslzshuff])

# %%
zscores,data,alphas = myload(find(f'alliaaftzscores_*0.2_0.7*',base_path+'signif/')[0])
num_points = data.shape[1]

emp_corr = compute_empcorr(data)
all_dens,all_fdr, all_fdrz,arbiash,arbiasl,arbiashz,arbiaslz = [np.zeros(len(denslist)) for _ in range(7)]

if os.path.exists(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt'):
    all_dens,all_fdr,all_fdrz,arbiash,arbiasl,arbiashz,arbiaslz = myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
else:
    for i, dens in enumerate(denslist):
        zscores[np.eye(num_points,dtype=bool)] = 0
        zscores = np.maximum(zscores.T,zscores)
        #print('Starting ', alpha)
        adjz = get_adj(zscores,dens,weighted=False)#np.maximum(all_quantadj[i],all_quantadj[i].T)
        adj = get_adj(emp_corr,dens,weighted=False)
        density = adjz.sum()/ (adjz.shape[0]*(adjz.shape[0]-1))
        all_dens[i] = density
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdr[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
        all_fdrz[i] = adjz[np.logical_and(adjz != 0, dists > maxdist)].sum() / adjz[adjz!=0].sum()
        deg = adj.sum(axis = 0)/(adj.shape[0]-1)
        degz = adjz.sum(axis=0)/(adjz.shape[0]-1)
        arbiash[i] = deg[ar_coeff != ar].mean() 
        arbiasl[i] = deg[ar_coeff == ar].mean()
        arbiashz[i] = degz[ar_coeff != ar].mean() 
        arbiaslz[i] = degz[ar_coeff == ar].mean()
    mysave(base_path+'signif/',f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt',[all_dens,all_fdr,all_fdrz,arbiash,arbiasl,arbiashz,arbiaslz])

# %%
# for filename in find(f'allarbiasquants_*ar0.2_0.7_time*',base_path+'signif/'):
#     print(filename)
#     all_fdr_shuff,all_fdrthres_shuff,all_dens_shuff,arbiasl_shuff,arbiash_shuff,tarbiasl_shuff,tarbiash_shuff = [np.zeros(len(alphas)) for _ in range(7)]
#     all_quantadj,data,alphas = myload(filename)
#     emp_corr = compute_empcorr(data)
    
#     for i, alpha in enumerate(alphas):
#         print('Starting ', alpha)
#         adj = np.maximum(all_quantadj[i],all_quantadj[i].T)
#         density = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
#         tadj = get_adj(emp_corr,density,weighted=False)
#         all_dens_shuff[i] = density
#         distadj = get_adj(5-dists, density,weighted=True)
#         maxdist = 5-distadj[distadj!=0].min()
#         all_fdr_shuff[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
#         all_fdrthres_shuff[i] = tadj[np.logical_and(tadj != 0, dists > maxdist)].sum() / tadj[tadj!=0].sum()
#         deg = adj.sum(axis = 0)/(adj.shape[0]-1)
#         tdeg = tadj.sum(axis=0)/(tadj.shape[0]-1)
#         arbiash_shuff[i] = deg[ar_coeff != ar].mean() 
#         arbiasl_shuff[i] = deg[ar_coeff == ar].mean()
#         tarbiash_shuff[i] = tdeg[ar_coeff != ar].mean() 
#         tarbiasl_shuff[i] = tdeg[ar_coeff == ar].mean()
#     mysave(base_path+'signif/', f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1],[all_dens_shuff, all_fdr_shuff, all_fdrthres_shuff, arbiash_shuff, arbiasl_shuff,tarbiash_shuff,tarbiasl_shuff,ar_coeff, alphas])








# %%
# how do zscores and quantiles differ? quantiles are better than zscores given same density, but do not reach optimal low net density!
zscoresl,datal,alphasl = myload(find(f'alliaaftzscores_*0.2_time*',base_path+'signif/')[0])
all_quantadjl,datal,alphasl = myload(find(f'alliaaft_arbiasquants*0.2_time*',base_path+'signif/')[0])

zscores,data,alphas = myload(find(f'alliaaftzscores_*0.2_0.7*',base_path+'signif/')[0])
all_quantadj,data,alphas = myload(find(f'alliaaft_arbiasquants*0.2_0.7*',base_path+'signif/')[0])


all_dens_shuff,all_fdr_quant,all_fdr_thres, all_fdr_z,all_dens_shuffl,all_fdr_quantl,all_fdr_thresl, all_fdr_zl,arbiash_thres,arbiash_quant,arbiash_z,arbiasl_thres,arbiasl_quant,arbiasl_z = [np.zeros(len(alphas)) for _ in range(14)]


print(np.all(data==data))
emp_corr = compute_empcorr(data)
emp_corrl = compute_empcorr(datal)
for i, alpha in enumerate(alphas):
    print('Starting ', alpha)
    adj = np.maximum(all_quantadj[i],all_quantadj[i].T)
    density = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
    zscores[np.eye(num_points,dtype=bool)] = 0
    zscores = np.maximum(zscores.T,zscores)
    adjl = np.maximum(all_quantadjl[i],all_quantadjl[i].T)
    densityl = adjl.sum()/ (adjl.shape[0]*(adjl.shape[0]-1))
    zscoresl[np.eye(num_points,dtype=bool)] = 0
    zscoresl = np.maximum(zscoresl.T,zscoresl)
    #print('Starting ', alpha)
    zadj = get_adj(zscores,density,weighted=False)
    tadj = get_adj(emp_corr,density,weighted=False)
    all_dens_shuff[i] = density
    distadj = get_adj(5-dists, density,weighted=True)
    maxdist = 5-distadj[distadj!=0].min()
    all_fdr_quant[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
    all_fdr_thres[i] = tadj[np.logical_and(tadj != 0, dists > maxdist)].sum() / tadj[tadj!=0].sum()
    all_fdr_z[i] = zadj[np.logical_and(zadj != 0, dists > maxdist)].sum() / zadj[zadj!=0].sum()
    zadjl = get_adj(zscoresl,densityl,weighted=False)
    tadjl = get_adj(emp_corrl,densityl,weighted=False)
    all_dens_shuffl[i] = densityl
    distadj = get_adj(5-dists, density,weighted=True)
    maxdist = 5-distadj[distadj!=0].min()
    all_fdr_quantl[i] = adjl[np.logical_and(adjl != 0, dists > maxdist)].sum() / adjl[adjl!=0].sum()
    all_fdr_thresl[i] = tadjl[np.logical_and(tadjl != 0, dists > maxdist)].sum() / tadjl[tadjl!=0].sum()
    all_fdr_zl[i] = zadjl[np.logical_and(zadjl != 0, dists > maxdist)].sum() / zadjl[zadjl!=0].sum()
    deg = adj.sum(axis = 0)/(adj.shape[0]-1)
    tdeg = tadj.sum(axis=0)/(tadj.shape[0]-1)
    zdeg = zadj.sum(axis=0)/(zadj.shape[0]-1)
    arbiash_quant[i] = deg[ar_coeff != ar].mean() 
    arbiasl_quant[i] = deg[ar_coeff == ar].mean()
    arbiash_thres[i] = tdeg[ar_coeff != ar].mean() 
    arbiasl_thres[i] = tdeg[ar_coeff == ar].mean()
    arbiash_z[i] = zdeg[ar_coeff != ar].mean() 
    arbiasl_z[i] = zdeg[ar_coeff == ar].mean()
    mysave(base_path+'signif/','all_iaaft_zvsq_stats_0.2_0.7.txt', [all_dens_shuff,all_fdr_quant,all_fdr_thres, all_fdr_z,arbiash_thres,arbiash_quant,arbiash_z,arbiasl_thres,arbiasl_quant,arbiasl_z])


# %%
long_dens,long_fdr,long_fdrz,longarbiash,longarbiasl,longarbiashz,longarbiaslz=myload(base_path+'signif/'+f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')
# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)
fig,ax = plt.subplots()
p1,=ax.plot(long_dens[1:], longarbiash[1:]/long_dens[1:], color = adjust_lightness('tab:grey', amount=1.5), label='t,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p2,=ax.plot(long_dens[1:], longarbiasl[1:]/long_dens[1:], color = adjust_lightness('tab:grey', amount=0.5), label='t,low AR')#, color = adjust_lightness('tab:blue', amount=0.5))
p3,=ax.plot(all_dens_shuff[1:], arbiash_quant[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:blue', amount=1.5), label='q,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p4,=ax.plot(all_dens_shuff[1:], arbiasl_quant[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:blue', amount=0.5), label='q,low AR')
p5,=ax.plot(long_dens[1:], longarbiashz[1:]/long_dens[1:], color = adjust_lightness('tab:orange', amount=1.5), label='z,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p6,=ax.plot(long_dens[1:], longarbiaslz[1:]/long_dens[1:], color = adjust_lightness('tab:orange', amount=0.5), label='z,low AR')
# p7,=ax.plot(all_densb[1:], ensarbiash2[1:], color = adjust_lightness('tab:red', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
# p8,=ax.plot(all_densb[1:], ensarbiasl2[1:], color = adjust_lightness('tab:red', amount=0.5), label='s,low AR')
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
#ax.set_xlim(-0.005,0.2)
l = ax.legend([(p1, p2),(p3,p4),(p5,p6)], ['threshold','quantiles','z scores'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
plt.savefig(base_path + f'arbias_iaaft_zvsq_normalized_relative_long_signif.pdf')


# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)
fig,ax = plt.subplots()
p1,=ax.plot(all_dens_shuff[1:], arbiash_thres[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:grey', amount=1.5), label='t,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p2,=ax.plot(all_dens_shuff[1:], arbiasl_thres[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:grey', amount=0.5), label='t,low AR')#, color = adjust_lightness('tab:blue', amount=0.5))
p5,=ax.plot(all_dens_shuff[1:], arbiash_z[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:orange', amount=1.5), label='z,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p6,=ax.plot(all_dens_shuff[1:], arbiasl_z[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:orange', amount=0.5), label='z,low AR')
p3,=ax.plot(all_dens_shuff[1:], arbiash_quant[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:blue', amount=1.5), label='q,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p4,=ax.plot(all_dens_shuff[1:], arbiasl_quant[1:]/all_dens_shuff[1:], color = adjust_lightness('tab:blue', amount=0.5), label='q,low AR')
# p7,=ax.plot(all_densb[1:], ensarbiash2[1:], color = adjust_lightness('tab:red', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
# p8,=ax.plot(all_densb[1:], ensarbiasl2[1:], color = adjust_lightness('tab:red', amount=0.5), label='s,low AR')
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
#ax.set_xlim(-0.005,0.2)
l = ax.legend([(p1, p2),(p3,p4),(p5,p6)], ['threshold','quantiles','z scores'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
plt.savefig(base_path + f'arbias_iaaft_zvsq_normalized_relative_signif.pdf')

# %%
adjust_fontsize(3)
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
fig,ax = plt.subplots()
ax.set_ylabel('FDR')
ax.set_xlabel('Density')
p1,=ax.plot(all_dens_shuffl[1:],all_fdr_zl[1:], color = adjust_lightness('tab:orange', amount=0.7), label='z,low')
p12,=ax.plot(all_dens_shuff[1:],all_fdr_z[1:], color = adjust_lightness('tab:orange', amount=1.3), label='z,mixed')
#p1,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='signif, high AR')
#p2,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.3), label='z, mixed AR')

#p2,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:orange', amount=1), label='signif, mixed AR')
#p3,=ax.plot(all_densh[1:],all_fdrhthres[1:], color = adjust_lightness('tab:blue', amount=1.5),label = 'thres, high AR')
p3,=ax.plot(all_dens_shuffl[1:],all_fdr_thresl[1:], color = adjust_lightness('tab:grey', amount=0.7),label = 'thres,l')
p34,=ax.plot(all_dens_shuff[1:],all_fdr_thres[1:], color = adjust_lightness('tab:grey', amount=1.3),label = 'thres,m')
#p4,=ax.plot(all_dens[1:],all_fdr[1:], color = adjust_lightness('tab:grey', amount=1.3), label='thres, mixed AR')

p5,=ax.plot(all_dens_shuffl[1:],all_fdr_quantl[1:], color = adjust_lightness('tab:blue', amount=0.7),label = 'quant,l')
p56,=ax.plot(all_dens_shuff[1:],all_fdr_quant[1:], color = adjust_lightness('tab:blue', amount=1.3),label = 'quant,m')
#p6,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:blue', amount=1.3), label='thres, mixed AR')
#p5,=ax.plot(all_densensh[1:],all_fdrens1h[1:], color = adjust_lightness('tab:green', amount=1.5),label = 'ens1, high AR')
#p6,=ax.plot(all_densens[1:],all_fdrens1[1:], color = adjust_lightness('tab:green', amount=1),label = 'ens1, mixed AR')
#p7,=ax.plot(all_densh[1:],all_fdrens2h[1:], color = adjust_lightness('tab:red', amount=1.5), label='ens2, high AR')
#p8,=ax.plot(all_densb[1:],all_fdrens2[1:], color = adjust_lightness('tab:red', amount=1), label='ens2, mixed AR')
l = ax.legend([(p3,p34),(p5,p56),(p1,p12)], ['threshold','quantiles','z scores'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False) ,(p5,p6),(p7,p8)] ,'Stab. selection','Bootstr. mean'
ax.set_xlim(0.002,0.1)
#plt.legend()
plt.savefig(base_path + f'fdr_iaaft_zvsq_signif3_allar.pdf')   #+ filename.split('allarbiasquants_',1)[1][:-4] + '.pdf')


# %%
fig,ax = plt.subplots()
ax.set_ylabel('FDR')
ax.set_xlabel('Density')
#p1,=ax.plot(all_dens_shuffl[1:],all_fdr_zl[1:], color = adjust_lightness('tab:orange', amount=0.7), label='z,low')
ax.plot(all_dens_shuff[1:],all_fdr_z[1:], color = adjust_lightness('tab:orange', amount=1), label='z scores')
#p1,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='signif, high AR')
#p2,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.3), label='z, mixed AR')

#p2,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:orange', amount=1), label='signif, mixed AR')
#p3,=ax.plot(all_densh[1:],all_fdrhthres[1:], color = adjust_lightness('tab:blue', amount=1.5),label = 'thres, high AR')
#p3,=ax.plot(all_dens_shuffl[1:],all_fdr_thresl[1:], color = adjust_lightness('tab:grey', amount=0.7),label = 'thres,l')
ax.plot(all_dens_shuff[1:],all_fdr_thres[1:], color = adjust_lightness('tab:grey', amount=1),label = 'threshold')
#p4,=ax.plot(all_dens[1:],all_fdr[1:], color = adjust_lightness('tab:grey', amount=1.3), label='thres, mixed AR')

#p5,=ax.plot(all_dens_shuffl[1:],all_fdr_quantl[1:], color = adjust_lightness('tab:blue', amount=0.7),label = 'quant,l')
ax.plot(all_dens_shuff[1:],all_fdr_quant[1:], color = adjust_lightness('tab:blue', amount=1),label = 'quantiles')
#p6,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:blue', amount=1.3), label='thres, mixed AR')
#p5,=ax.plot(all_densensh[1:],all_fdrens1h[1:], color = adjust_lightness('tab:green', amount=1.5),label = 'ens1, high AR')
#p6,=ax.plot(all_densens[1:],all_fdrens1[1:], color = adjust_lightness('tab:green', amount=1),label = 'ens1, mixed AR')
#p7,=ax.plot(all_densh[1:],all_fdrens2h[1:], color = adjust_lightness('tab:red', amount=1.5), label='ens2, high AR')
#p8,=ax.plot(all_densb[1:],all_fdrens2[1:], color = adjust_lightness('tab:red', amount=1), label='ens2, mixed AR')
#l = ax.legend([(p3,p34),(p5,p56),(p1,p12)], ['threshold','quantiles','z scores'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False) ,(p5,p6),(p7,p8)] ,'Stab. selection','Bootstr. mean'
ax.set_xlim(0.002,0.1)
plt.legend()
plt.savefig(base_path + f'fdr_iaaft_zvsq_signif3_mixedar.pdf')   #+ filename.split('allarbiasquants_',1)[1][:-4] + '.pdf')

# %%
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
plt.rcParams.update(bundles.icml2022())
adjust_fontsize(3)
fig,ax = plt.subplots()
p1,=ax.plot(all_dens[1:], arbiash[1:], color = adjust_lightness('tab:grey', amount=1.5), label='t,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p2,=ax.plot(all_dens[1:], arbiasl[1:], color = adjust_lightness('tab:grey', amount=0.5), label='t,low AR')#, color = adjust_lightness('tab:blue', amount=0.5))
p5,=ax.plot(all_dens[1:], arbiashz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='iaaft,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p6,=ax.plot(all_dens[1:], arbiaslz[1:], color = adjust_lightness('tab:orange', amount=0.5), label='iaaft,low AR')
p3,=ax.plot(all_dens_shuff[1:], arbiash_shuff[1:], color = adjust_lightness('tab:blue', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
p4,=ax.plot(all_dens_shuff[1:], arbiasl_shuff[1:], color = adjust_lightness('tab:blue', amount=0.5), label='s,low AR')
# p7,=ax.plot(all_densb[1:], ensarbiash2[1:], color = adjust_lightness('tab:red', amount=1.5), label='s,high AR')#, color = adjust_lightness('tab:blue', amount=1.5))
# p8,=ax.plot(all_densb[1:], ensarbiasl2[1:], color = adjust_lightness('tab:red', amount=0.5), label='s,low AR')
ax.set_xlabel('Density')
ax.set_ylabel('Avg. norm. degree')
#ax.set_xlim(-0.005,0.2)
l = ax.legend([(p1, p2),(p3,p4),(p5,p6)], ['threshold','shuffles','IAAFT'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False)
plt.savefig(base_path + f'arbias_iaaft_normalized_signif.pdf')

# %%
for filename in find(f'allarbiasquants_*ar0.2_time*',base_path+'signif/'):
    print(filename)
    all_fdr_low,all_fdrthres_low,all_dens_low = [np.zeros(len(alphas)) for _ in range(3)]
    all_quantadj,data,alphas = myload(filename)
    emp_corr = compute_empcorr(data)
    
    for i, alpha in enumerate(alphas):
        print('Starting ', alpha)
        adj = np.maximum(all_quantadj[i],all_quantadj[i].T)
        density = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
        tadj = get_adj(emp_corr,density,weighted=False)
        all_dens_low[i] = density
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdr_low[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
        all_fdrthres_low[i] = tadj[np.logical_and(tadj != 0, dists > maxdist)].sum() / tadj[tadj!=0].sum()
        
    mysave(base_path+'signif/', f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1],[all_dens_low, all_fdr_low, all_fdrthres_low, alphas])

# %%
for filename in find(f'allarbiasquants_*ar0.7_time*',base_path+'signif/'):
    print(filename)
    all_fdr_high,all_fdrthres_high,all_dens_high = [np.zeros(len(alphas)) for _ in range(3)]
    all_quantadj,data,alphas = myload(filename)
    emp_corr = compute_empcorr(data)
    
    for i, alpha in enumerate(alphas):
        print('Starting ', alpha)
        adj = np.maximum(all_quantadj[i],all_quantadj[i].T)
        density = adj.sum()/ (adj.shape[0]*(adj.shape[0]-1))
        tadj = get_adj(emp_corr,density,weighted=False)
        all_dens_high[i] = density
        distadj = get_adj(5-dists, density,weighted=True)
        maxdist = 5-distadj[distadj!=0].min()
        all_fdr_high[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
        all_fdrthres_high[i] = tadj[np.logical_and(tadj != 0, dists > maxdist)].sum() / tadj[tadj!=0].sum()
        
    mysave(base_path+'signif/', f'arbiasstats_'+ filename.split('allarbiasquants_',1)[1],[all_dens_high, all_fdr_high, all_fdrthres_high, alphas])

# %%
zscores,data,alphas = myload(find(f'alliaaftzscores_*0.2_time*',base_path+'signif/')[0])
num_points = data.shape[1]

emp_corr = compute_empcorr(data)
all_dens,all_fdr_lowiaaft, all_fdrz_lowiaaft = [np.zeros(len(denslist)) for _ in range(3)]

for i, dens in enumerate(denslist):
    zscores[np.eye(num_points,dtype=bool)] = 0
    zscores = np.maximum(zscores.T,zscores)
    #print('Starting ', alpha)
    adjz = get_adj(zscores,dens,weighted=False)#np.maximum(all_quantadj[i],all_quantadj[i].T)
    adj = get_adj(emp_corr,dens,weighted=False)
    density = adjz.sum()/ (adjz.shape[0]*(adjz.shape[0]-1))
    all_dens[i] = density
    distadj = get_adj(5-dists, density,weighted=True)
    maxdist = 5-distadj[distadj!=0].min()
    all_fdr_lowiaaft[i] = adj[np.logical_and(adj != 0, dists > maxdist)].sum() / adj[adj!=0].sum()
    all_fdrz_lowiaaft[i] = adjz[np.logical_and(adjz != 0, dists > maxdist)].sum() / adjz[adjz!=0].sum()
    
mysave(base_path+'signif/',f'all_iaaftstats_30_nu1.5_len0.2_ar0.2_time100.txt',[all_dens,all_fdr_lowiaaft,all_fdrz_lowiaaft])

# %%
#all_dens, all_fdr, all_fdrthres, arbiash, arbiasl,tarbiash,tarbiasl,ar_coeff, alphas= myload(base_path + 'signif/arbiasstats_30_nu1.5_len0.2_ar0.2_time100.txt')
#all_densshuff, all_fdrshuff,all_fdrthres,_, alphas = myload(base_path + 'signif/arbiasstats_30_nu1.5_len0.2_ar0.2_time100.txt')
#all_densb, all_fdrbshuff, all_fdrbthres, arbiashshuff, arbiaslshuff,tarbiash,tarbiasl, ar_coeff, alphas = myload(base_path + 'signif/arbiasstats_30_nu1.5_len0.2_ar0.2_0.7_time100.txt')

# %%
ars = autocorr(data)

plt.hist(np.diag(ars))
# %%
(zscores == -9999).mean()






# %%
# add high AR
adjust_fontsize(3)
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
fig,ax = plt.subplots()
ax.set_ylabel('FDR')
ax.set_xlabel('Density')
p12,=ax.plot(all_dens[1:],all_fdrz_lowiaaft[1:], color = adjust_lightness('tab:orange', amount=0.7), label='signif, low AR')
#p1,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.5), label='signif, high AR')
p2,=ax.plot(all_dens[1:],all_fdrz[1:], color = adjust_lightness('tab:orange', amount=1.3), label='signif, mixed AR')

#p2,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:orange', amount=1), label='signif, mixed AR')
#p3,=ax.plot(all_densh[1:],all_fdrhthres[1:], color = adjust_lightness('tab:blue', amount=1.5),label = 'thres, high AR')
p34,=ax.plot(all_dens[1:],all_fdr_lowiaaft[1:], color = adjust_lightness('tab:grey', amount=0.7),label = 'thres, low AR')
p4,=ax.plot(all_dens[1:],all_fdr[1:], color = adjust_lightness('tab:grey', amount=1.3), label='thres, mixed AR')

p56,=ax.plot(all_dens_low[1:],all_fdr_low[1:], color = adjust_lightness('tab:blue', amount=0.7),label = 'thres, low AR')
p6,=ax.plot(all_dens_shuff[1:],all_fdr_shuff[1:], color = adjust_lightness('tab:blue', amount=1.3), label='thres, mixed AR')
#p5,=ax.plot(all_densensh[1:],all_fdrens1h[1:], color = adjust_lightness('tab:green', amount=1.5),label = 'ens1, high AR')
#p6,=ax.plot(all_densens[1:],all_fdrens1[1:], color = adjust_lightness('tab:green', amount=1),label = 'ens1, mixed AR')
#p7,=ax.plot(all_densh[1:],all_fdrens2h[1:], color = adjust_lightness('tab:red', amount=1.5), label='ens2, high AR')
#p8,=ax.plot(all_densb[1:],all_fdrens2[1:], color = adjust_lightness('tab:red', amount=1), label='ens2, mixed AR')
l = ax.legend([(p4,p34),(p6,p56),(p2,p12)], ['threshold','shuffles','IAAFT'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}) #frameon=False) ,(p5,p6),(p7,p8)] ,'Stab. selection','Bootstr. mean'
ax.set_xlim(0.002,0.1)
plt.savefig(base_path + f'fdr_iaaft_all_signif3_'+ filename.split('allarbiasquants_',1)[1][:-4] + '.pdf')
# %%
# what happens to highest IAAFT zscores: check means and std
density = 0.005
distadj = get_adj(5-dists, density,weighted=True)
maxdist = 5-distadj[distadj!=0].min()

import tqdm
ac = 0.2
ac2 = 0.7
ar_coeffidcs = myload(base_path + f'arcoeffidcs_{num_points}.txt')
ar_coeff = ac * np.ones(num_points)
ar_coeff[ar_coeffidcs] = ac2
n_time=100
n_lat = 4 * 18
num_points = gridstep_to_numpoints(180/n_lat)
#quants,ar_coeff,alphas = myload(find('arbiasquants_iaaft_part0_*',base_path+'signif/')[0])
gslist = []
for filename in find('arbiasquants_iaaft_part0_*0.2_0.7*', base_path+ 'signif/',nosub=True):
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
    highmeans = []
    highstd = []
    mixedmeans = []
    mixedstd = []
    lowmeans = []
    lowstd = []
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
        randidcs = np.random.random(len(keys))
        for i in range(len(keys)):
            #if randidcs[i]< density: random edges
            #if dists[keys[i][0], keys[i][1]] < maxdist: high true corr
            if keys[i][0] in ar_coeffidcs:
                if keys[i][1] in ar_coeffidcs:
                    highmeans.append(mean_std[keys[i]][0])
                    highstd.append(mean_std[keys[i]][1])
                else:
                    mixedmeans.append(mean_std[keys[i]][0])
                    mixedstd.append(mean_std[keys[i]][1])
            else:
                if keys[i][1] in ar_coeffidcs:
                    mixedmeans.append(mean_std[keys[i]][0])
                    mixedstd.append(mean_std[keys[i]][1])
                else:
                    lowmeans.append(mean_std[keys[i]][0])
                    lowstd.append(mean_std[keys[i]][1])

            
                
        
        # for ialpha in range(len(alphas)):
        #     for i in range(len(keys)):
        #         all_quantadj[ialpha][keys[i][0], keys[i][1]] = (emp_corr[keys[i][0], keys[i][1]] > quants[keys[i]][ialpha])
        #     all_quantadj[ialpha]=np.maximum(all_quantadj[ialpha].T,all_quantadj[ialpha])
# %%
plt.hist(highmeans,50)
# %%
plt.hist(lowmeans,50)
# %%
plt.hist(mixedmeans,50)
# %%
plt.hist(highmeans,50)
plt.xlabel('IAAFT means of high z scores')
plt.savefig(base_path+'iaaftmeans_highz.pdf',dpi=300)

# %%
plt.hist(highstd,50)
plt.xlabel('IAAFT stds of high z scores')
plt.savefig(base_path+'iaaftstds_highz.pdf',dpi=300)

# %%
varbins = np.linspace(0.05,0.3,100)
lowstdhist = np.histogram(lowstd,varbins)[0]
highstdhist = np.histogram(highstd,varbins)[0]
lowstdhist = lowstdhist/lowstdhist.sum()
highstdhist = highstdhist/highstdhist.sum()

meanbins = np.linspace(-0.02,0.02,100)
highmeanhist = np.histogram(highmeans,meanbins)[0]
lowmeanhist = np.histogram(lowmeans,meanbins)[0]
lowmeanhist = lowmeanhist/lowmeanhist.sum()
highmeanhist = highmeanhist/highmeanhist.sum()

# %%
plt.axhline(0,color = 'black')
plt.plot(meanbins[1:],highmeanhist-lowmeanhist)
plt.xlabel('IAAFT Mean Estimate')
plt.ylabel('High AR - Low AR Histogram')
plt.savefig(base_path+'delta_iaaftmeans_acrossAR.pdf',dpi=300)

# %%
plt.axhline(0,color = 'black')
plt.plot(varbins[1:],highstdhist-lowstdhist)
plt.xlabel('IAAFT Std Estimate')
plt.ylabel('High AR - Low AR Histogram')
plt.savefig(base_path+'delta_iaaftstds_acrossAR.pdf',dpi=300)


# %%
varbins = np.linspace(0.05,0.35,50)
normalstdhist = np.histogram(normalstd,varbins)[0]
highstdhist = np.histogram(highstd,varbins)[0]
normalstdhist = normalstdhist/normalstdhist.sum()
highstdhist = highstdhist/highstdhist.sum()

meanbins = np.linspace(-0.02,0.02,50)
normalmeanhist = np.histogram(normalmeans,meanbins)[0]
highmeanhist = np.histogram(highmeans,meanbins)[0]
normalmeanhist = normalmeanhist/normalmeanhist.sum()
highmeanhist = highmeanhist/highmeanhist.sum()

# %%
plt.plot(meanbins[1:],highmeanhist-normalmeanhist)
plt.xlabel('IAAFT Mean Estimate')
plt.ylabel('High vs Random zscores')
plt.savefig(base_path+'delta_iaaftmeans_highz.pdf',dpi=300)


# %%
plt.plot(varbins[1:],highstdhist-normalstdhist)
plt.xlabel('IAAFT Std Estimate')
plt.ylabel('High vs Random zscores')
plt.savefig(base_path+'delta_iaaftstds_highz.pdf',dpi=300)

# So high IAAFT zscores do not have vanishing variance estimates,
# but larger mean estimate variance
