# %%
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
#from numpy.lib.type_check import nan_to_num
from scipy import signal, stats
# import cartopy as ctp
import xarray as xr
from climnet.dataset_new import AnomalyDataset
from climnet.grid import regular_lon_lat, FeketeGrid
from climnet.myutils import *
from climnet.similarity_measures import *
import time
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()
curr_time = time.time()
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})

print('HOME is: ', os.getenv("HOME"))
distrib = 'igrf'

grid_type = 'fekete'
n_lat = 18*4
#typ ='threshold' # iterated over
corr_method='pearson'
#weighted = False # iterate over
ranks = False
if corr_method in ['BI-KSG', 'binMI', 'HSIC']:
    ranks = True

#var_name = 't2m' # ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr'] #, 'sp'
var_names = ['t2m', 't2mdaily', 'z250', 'z500', 'z850', 'pr', 'sp']

denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
# ks = [6, 60, 300, 600,1200]

#filter_string = f'_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_'


#grid_helper and calc_true
num_runs = 2
n_time = 100
nus = [0.5,1.5]
len_scales = [0.1,0.2]
nu = 0.5
len_scale = 0.1
ar=0
ar2=None
var=10
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
# %%
n_boot = 10
weighted = False
denslist = [0.001,0.0025, 0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.1,0.2]#[0.001,0.01,0.05,0.1,0.2]
seed = 1204
timeparts = 10
density = 0.01
# Grid
grid_step = 180 / n_lat
alpha=0.05
rm_outliers = True
#num_cpus = 48
#lon_range = [-110, 40]
#lat_range = [20, 85]
save = True
# time_range = ['1980-01-01', '2019-12-31']
time_range = None
tm = None  # 'week'
norm = False
#detrend = True
absolute = False

PATH = os.path.dirname(os.path.abspath(__file__))
print('You are here: ', PATH)


base_path = '../../climnet_output/'
base_path = base_path + 'rfstats/'
orig_base_path = '../../climnet_output/'
if not os.path.exists(base_path):
    os.mkdir(base_path)

threshs = [0.1,0.2,0.3,0.5,0.8]
all_dens = np.zeros((len(nus),len(len_scales),len(threshs)))

lenbins = np.linspace(0,np.pi,100)
denslistplus = deepcopy(denslist)
denslistplus.append(density)
    
dens_diff = np.zeros((len(nus),len(len_scales), len(denslistplus),n_boot))
lldistrib = np.zeros((len(nus),len(len_scales), len(denslistplus),2*n_boot,len(lenbins)-1))
difflldistrib = np.zeros((len(nus),len(len_scales), len(denslistplus),n_boot,len(lenbins)-1))

dtele = 5000 / earth_radius # what is a teleconnection? Longer than 5000 km is a cautious choice

shortdenslist = [0.001,0.01,0.05,0.1,0.2]

#all_lengths1 = np.zeros((len(shortdenslist),n_boot,len(dists)))
#all_lengths2 = np.zeros((len(shortdenslist),n_boot,len(dists)))
all_countrobusttele2raw1, all_countrobusttele2raw2, all_thresh,all_teledens,all_countdifftele2raw1,all_countdifftele2raw2 = [np.zeros((len(nus),len(len_scales),len(shortdenslist),n_boot)) for _ in range(6)]

curr_time = time.time()
for inu,nu in enumerate(nus):
    for ilen, len_scale in enumerate(len_scales):
        if corr_method == 'BI-KSG':
            empcorrs = [nam for nam in find(f'empcorrdict_*_BI-KSG_matern_nu{nu}_len{len_scale}_*',orig_base_path+'empcorrs/') if not fnmatch.fnmatch(nam,'*_part*')]
        else:
            empcorrs = find(f'{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',orig_base_path+'empcorrs/')
        empdatas = find(f'*_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_*',orig_base_path+'empdata/')
        #for w in [True,False]:
    
        # if typ == 'knn' and not w:
        #     continue
        #outfilename = f'rfstats_part{irun}_{distrib}_{corr_method}_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_{num_runs}.txt'
        all_degs, all_lls,all_degws, all_llws, all_ccs,all_ccws, all_spls = [np.zeros((numth, num_bins)) for _ in range(7)]
        all_llquant1,all_llquant2,all_llwquant1,all_llwquant2 = [np.zeros((numth)) for _ in range(4)]
        for irun in range(n_boot):
            if 2*irun+1 < len(empcorrs):
                emp_corr = myload(empcorrs[2*irun])
                emp_corr2 = myload(empcorrs[2*irun+1])
            else:
                if 2*irun+1 < len(empdatas):
                    orig_data = myload(empdatas[2*irun])
                    orig_data2 = myload(empdatas[2*irun+1])
                else:
                    seed = int(time.time())
                    np.random.seed(seed)
                    orig_data = diag_var_process(ar_coeff, cov, n_time)
                    mysave(orig_base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',orig_data)
                    seed = int(time.time())
                    np.random.seed(seed)
                    orig_data2 = diag_var_process(ar_coeff, cov, n_time)
                    mysave(orig_base_path+'empdata/', f'data_matern_nu{nu}_len{len_scale}_ar{ar}_{grid_type}{n_lat}_time{n_time}_var1_seed{seed}.txt',orig_data)
                data = np.sqrt(var) * orig_data
                data2 = np.sqrt(var) * orig_data2
                # save with nam in name
                for j in range(len(lat)):
                    data[:,j] -= orig_data[:,j].mean()
                    data[:,j] /= orig_data[:,j].std()
                    data2[:,j] -= orig_data2[:,j].mean()
                    data2[:,j] /= orig_data2[:,j].std()
                if corr_method == 'BI-KSG':
                    raise FileNotFoundError(f'{irun}, outfilename')
                else:
                    emp_corr = compute_empcorr(data, similarity=corr_method)
                    emp_corr2 = compute_empcorr(data2, similarity=corr_method)
        

            # bootstrap data in time
            for idens,dens in enumerate(shortdenslist):
                adj1 = get_adj(emp_corr,dens= dens,weighted=weighted)
                thresh = np.abs(emp_corr)[adj1!=0].min()
                all_thresh[inu,ilen,idens,irun] = thresh
                # distadj = get_adj(5-dists, density,weighted=True)
                # maxdist = 5-distadj[distadj!=0].min()
                #all_lengths1[inu,ilen,idens,irun,:] = decorr_length(adj1,dists,min_connectivity=0.8, grid_type=grid_type,base_path=base_path)#bin then save
                #all_lengths2[inu,ilen,idens,irun,:] = decorr_length(adj1,dists,min_connectivity=0.5, grid_type=grid_type,base_path=base_path)
                # how many telelinks in bundles?
                teleadj = deepcopy(adj1)
                teleadj[dists < dtele] = 0
                all_teledens[inu,ilen,idens,irun] = teleadj.sum()/((adj1.shape[0]-1)*adj1.shape[0])
                if teleadj.sum() == 0:
                    all_countrobusttele2raw1[inu,ilen,idens,irun] = 0
                    all_countrobusttele2raw2[inu,ilen,idens,irun] = 0
                else:
                    all_countrobusttele2raw1[inu,ilen,idens,irun] =  bundlefraction(adj1, dists, nbhds2, dtele, tolerance = 0.2, typ = 'raw')
                    all_countrobusttele2raw2[inu,ilen,idens,irun] =  bundlefraction(adj1, dists, nbhds2, dtele, tolerance = 0.5, typ = 'raw')
                adj2 = get_adj(emp_corr2, dens= dens,weighted=weighted)
                diff = np.where(np.logical_and(np.abs(adj1 - adj2)!=0,dists>dtele))
                # compute how many differing links are part of some bundle
                all_countdifftele2raw1[inu,ilen,idens,irun] = bundlefractionwhere(adj1, diff, nbhds2, tolerance = 0.2, typ = 'raw')
                all_countdifftele2raw2[inu,ilen,idens,irun] = bundlefractionwhere(adj1, diff, nbhds2, tolerance = 0.5, typ = 'raw')
            

mysave(base_path,f'decorrstats_perdens_sim_dtele{dtele}_{corr_method}_w{weighted}_{n_boot}.txt', [all_thresh,all_teledens,all_countrobusttele2raw1,all_countrobusttele2raw2,all_countdifftele2raw1,all_countdifftele2raw2])



# %%