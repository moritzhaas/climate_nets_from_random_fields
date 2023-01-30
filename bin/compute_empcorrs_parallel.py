# %%
import os, pickle
import numpy as np
from climnet.grid import regular_lon_lat, FeketeGrid
from climnet.myutils import *
from climnet.event_synchronization import event_sync, event_synchronization, event_synchronization_matrix
from climnet.similarity_measures import *
from scipy import stats
import time
#from multiprocessing import Pool
import fnmatch
itask = int(os.environ['SLURM_ARRAY_TASK_ID'])
print('Task: ', itask)
curr_time = time.time()

# compute one matrix after the other instead
def compute_edges(data, mtx_idcs,mtx_idcsj,q_mi=2):
    if len(mtx_idcs) != len(mtx_idcsj):
        raise ValueError(f'Incompatible index lengths: {len(mtx_idcs)} != {len(mtx_idcsj)}')
    empcorr_dict = {}
    for idx in range(len(mtx_idcs)):
        i,j = mtx_idcs[idx], mtx_idcsj[idx]
        #print(i,j)
        empcorr_dict[(i,j)] = revised_mi(data[:,i], data[:,j], q = q_mi)
        #print( [quantile(shufflecorrs,alpha = alpha) for alpha in alphas])
    return empcorr_dict

def save_empcorrs(nam,similarity):
    orig_data = myload(nam)
    nam = nam.split('data_',1)[1]
    data = np.sqrt(var) * orig_data
    exp_data = np.exp(np.sqrt(var) * orig_data)
    # save with nam in name
    for j in range(len(lat)):
        data[:,j] -= orig_data[:,j].mean()
        data[:,j] /= orig_data[:,j].std()
        data2[:,j] -= exp_data[:,j].mean()
        data2[:,j] /= exp_data[:,j].std()
    
    emp_corr = compute_empcorr(data, similarity)
    mysave(base_path+f'empcorrs/',f'igrf_{similarity}_'+nam,emp_corr)

    emp_corr = compute_empcorr(data2, similarity)
    mysave(base_path+f'empcorrs/',f'expigrf_{similarity}_'+nam,emp_corr)
    print('Done:', similarity, nam)
    return emp_corr

if __name__ == '__main__':
    #number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

    #plt.rcParams["figure.figsize"] = (8*2,6*2)
    #plt.style.use('bmh')

    # Grid
    num_runs = 30
    num_tasks = 30
    filter_string = '*ar0_*'
    n_time = 100
    n_lat = 18 * 4
    grid_type = 'fekete'
    ar = 0
    var = 10
    alpha1 = 0.05

    base_path = '../../climnet_output/' #'~/Documents/climnet_output/'
    #name = f'{distrib}_{corr_method}'
    #filename = base_path + name + f'_{typ}_gamma{gamma}_K{K}_ntime{n_time}_nlat{n_lat}_{num_runs}runs_var{var}_fullrange.nc'


    # generate grid
    n_lon = 2 * n_lat
    grid_step_lon = 360/ n_lon
    grid_step_lat = 180/ n_lat
    dist_equator = gdistance((0,0),(0,grid_step_lon),radius=1)
    lon2, lat2 = regular_lon_lat(n_lon,n_lat)
    regular_grid = {'lon': lon2, 'lat': lat2}
    start_date = '2000-01-01'
    reg_dists = myload(base_path +f'grids/regular_dists_nlat_{n_lat}_nlon_{n_lon}.txt')

    # create fekete grid
    num_points = gridstep_to_numpoints(grid_step_lon)
    grid = FeketeGrid(num_points = num_points)
    lon, lat = grid.grid['lon'], grid.grid['lat']
    dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')

    earth_radius = 6371.009
    num_nodes = dists.shape[0]

    eps2 = 2 * dist_equator
    eps3 = 3 * dist_equator
    alpha1 = 0.95
    alpha2 = 0.99

    seed = int(time.time())
    np.random.seed(seed)
    # generate igrf data
    data = np.zeros((n_time,num_points))
    data2 = np.zeros((n_time,num_points))

    nu = 0.5
    len_scale = 0.1
    for similarity in ['BI-KSG', 'ES', 'HSIC','binMI']: #'pearson', 'spearman', 'binMI'
        for irun in range(0,num_runs):
            filter_string = f'*matern_nu{nu}_len{len_scale}_ar0_fekete{n_lat}_*'
            if find(filter_string, base_path+ 'empdata/') == []:
                raise NameError(filter_string + ' not in empdata')
            else:
                args = []
                for nam in find(filter_string, base_path+ 'empdata/'):
                    if not fnmatch.fnmatch(nam, '*_len0.0*'):
                        args.append(nam)
                print(filter_string, len(args),len(find(filter_string, base_path+ 'empdata/')))
                orig_data = myload(args[irun])
                nam = nam.split('data_',1)[1]
                if os.path.exists(base_path+f'empcorrs/empcorrdict_run{irun}_part{itask}_{similarity}_'+nam):
                    continue
                data = np.sqrt(var) * orig_data
                exp_data = np.exp(np.sqrt(var) * orig_data)
                # save with nam in name
                for j in range(len(lat)):
                    data[:,j] -= orig_data[:,j].mean()
                    data[:,j] /= orig_data[:,j].std()
                    data2[:,j] -= exp_data[:,j].mean()
                    data2[:,j] /= exp_data[:,j].std()
                try:
                    mtx_idcs
                except:
                    mtx_idcs = np.triu_indices_from(np.corrcoef(data.T),k=1)
                    num_edges = len(mtx_idcs[0])
                for i in range(num_tasks):
                    args.append((data, mtx_idcs[0][int(i*num_edges/num_tasks):int((i+1)*num_edges/num_tasks)],mtx_idcs[1][int(i*num_edges/num_tasks):int((i+1)*num_edges/num_tasks)]))
                empcorr_dict = compute_edges(data, mtx_idcs[0][int(itask*num_edges/num_tasks):int((itask+1)*num_edges/num_tasks)],mtx_idcs[1][int(itask*num_edges/num_tasks):int((itask+1)*num_edges/num_tasks)])
                mysave(base_path+f'empcorrs/',f'empcorrdict_run{irun}_part{itask}_{similarity}_'+nam,empcorr_dict)

                #emp_corr = compute_empcorr(data2, similarity)
                #mysave(base_path+f'empcorrs/',f'expigrf_{similarity}_'+nam,emp_corr)
                print('Done:', similarity, irun)



# %%
# from joblib import Parallel, delayed
# import multiprocessing as mpi
# from tqdm import tqdm
# num_cpus=mpi.cpu_count()

# #for job array on slurm cluster
# try:
#     min_job_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
#     max_job_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
#     job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#     num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
#     num_jobs = max_job_id
#     print(f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}" )

# except KeyError:
#     job_id = 0
#     num_jobs = 1
#     print("Not running with SLURM job arrays, but with manual id: ", job_id)

# diff_links = len(link_numbers)
# one_array_length = int(diff_links/num_jobs) + 1

# start_arr_idx = job_id * one_array_length
# end_arr_idx = (job_id + 1) * one_array_length

# # For parallel Programming
# print(f"Number of available CPUs: {num_cpus} for link bundeling!")
# print(f"Number of different number of links {len(link_numbers)}.")
# backend = 'multiprocessing'
# (Parallel(n_jobs=num_cpus, backend=backend)
#             (delayed(link_bundle_null_model_link_number)
#             (coord_rad, num_links,
#             link_bundle_folder, filename, bw, num_rand_permutations)
#             for num_links in tqdm(link_numbers[start_arr_idx:end_arr_idx]))
#     )


# res
# %%
    # # %%
    # for nam in find(f'data_matern_*_{grid_type}{n_lat}_*', base_path+ 'empdata/'):
    #     # compute and save empcorrs for all similarity estimators and for exp
    #     orig_data = myload(nam)
    #     nam = nam.split('data_',1)[1]
    #     data = orig_data
    #     exp_data = np.exp(orig_data)
    #     # save with nam in name
    #     for j in range(len(lat)):
    #         data[:,j] -= orig_data[:,j].mean()
    #         data[:,j] /= orig_data[:,j].std()
    #         data2[:,j] -= exp_data[:,j].mean()
    #         data2[:,j] /= exp_data[:,j].std()
        
    #     emp_corr = np.corrcoef(data.T)
    #     mysave(base_path+'empcorrs/','igrf_pearson_'+nam,emp_corr)
    #     emp_corr = np.corrcoef(data2.T)
    #     mysave(base_path+'empcorrs/','expigrf_pearson_'+nam,emp_corr)
        
    #     emp_corr, pval = stats.spearmanr(data)
    #     mysave(base_path+'empcorrs/','igrf_spearman_'+nam,emp_corr)

    #     for i in range(data.shape[1]):
    #         for j in range(i):
    #             emp_corr[i,j] = revised_mi(data[:,i], data[:,j], q = 2)
    #             emp_corr[j,i] = emp_corr[i,j]
    #     print("Maximal MI is ", emp_corr.max())
    #     mysave(base_path+'empcorrs/','igrf_BIKSG_'+nam,emp_corr)

    #     for i in range(data.shape[1]):
    #         for j in range(i):
    #             emp_corr[i,j] = revised_mi(data2[:,i], data2[:,j], q = 2)
    #             emp_corr[j,i] = emp_corr[i,j]
    #     print("Maximal MI is ", emp_corr.max())
    #     mysave(base_path+'empcorrs/','expigrf_BIKSG_'+nam,emp_corr)

    #     emp_corr = calculate_mutual_information(data, n_bins=np.floor(np.sqrt(n_time/5)))
    #     mysave(base_path+'empcorrs/','igrf_binMI_'+nam,emp_corr)
    #     emp_corr = calculate_mutual_information(data2, n_bins=np.floor(np.sqrt(n_time/5)))
    #     mysave(base_path+'empcorrs/','expigrf_binMI_'+nam,emp_corr)

    #     for i in range(data.shape[1]):
    #         for j in range(i):
    #             hsic = HSIC(data[:,i], data[:,j],unbiased=False, permute = False)
    #             emp_corr[i,j] = hsic
    #             emp_corr[j,i] = emp_corr[i,j]
    #     print("Maximal HSIC is ", emp_corr.max())
    #     mysave(base_path+'empcorrs/','igrf_HSIC_'+nam,emp_corr)
    #     for i in range(data.shape[1]):
    #         for j in range(i):
    #             hsic = HSIC(data2[:,i], data2[:,j],unbiased=False, permute = False)
    #             emp_corr[i,j] = hsic
    #             emp_corr[j,i] = emp_corr[i,j]
    #     print("Maximal HSIC is ", emp_corr.max())
    #     mysave(base_path+'empcorrs/','expigrf_HSIC_'+nam,emp_corr)
        
    #     sorted_dat = np.sort(np.abs(data),axis=0)
    #     dat_quantiles = sorted_dat[int(np.ceil(alpha1 * len(sorted_dat)-1)),:]
    #     events= (np.abs(data).T >= np.repeat(dat_quantiles.reshape((-1,1)),n_time, axis = 1))
    #     es = event_synchronization_matrix(events)
    #     mysave(base_path+'empcorrs/','igrf_ES_'+nam,es)
    #     print("Maximal ES is ", es.max())
    #     sorted_dat = np.sort(np.abs(data2),axis=0)
    #     dat_quantiles = sorted_dat[int(np.ceil(alpha1 * len(sorted_dat)-1)),:]
    #     events= (np.abs(data2).T >= np.repeat(dat_quantiles.reshape((-1,1)),n_time, axis = 1))
    #     es = event_synchronization_matrix(events)
    #     mysave(base_path+'empcorrs/','expigrf_ES_'+nam,es)
    #     # compute smth common to all event synch?
    #     #emp_corr = event_synchronization(data)
    #     print("Maximal ES is ", es.max())
    # # %%
    # import numpy as np

    # a = np.random.rand(20).reshape((4,5))
    # # %%
