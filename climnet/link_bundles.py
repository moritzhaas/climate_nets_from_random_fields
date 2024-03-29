#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:33:56 2021

@author: Felix Strnad
"""

# python libraries
import numpy as np
import sys, os
import xarray as xr
from sklearn.neighbors import KernelDensity
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mpi
from tqdm import tqdm
from climnet.utils import temp_seed





def spherical_kde(link_points, coord_rad, bw_opt):
    """
    Inspired from https://science.nu/amne/in-depth-kernel-density-estimation/
    Because the coordinate system here lies on a spherical surface rather than a flat plane, we will use the haversine distance metric,
       which will correctly represent distances on a curved surface.

    Parameters
    ----------
    link_points: np.array (num_links, 2)
        List of latidude and longitudes.
    coord_rad : array 
        Array of all links provided as [lon, lat]
    bw_opt: float
        bandwidth of the kde, used Scott rule here

    Returns
    -------
    Link density estimation 

    """
    assert link_points.shape[1] == 2
    # Do KDE fit by using haversine metric that accounts for spherical coordinates
    kde = KernelDensity(metric='haversine', kernel='gaussian', 
                        algorithm='ball_tree',bandwidth=bw_opt)
    kde.fit(link_points)
    Z = np.exp(kde.score_samples(coord_rad))

    return Z


def hist_approx(link_points, coord_rad):
    """Approximate spherical kde. 
    
    # TODO: This doesn't work yet. Recheck! 
    """

    assert link_points.shape[1] == 2
    binrule = 'sturges'
    p_lat, be_lat = np.histogram(link_points[:, 0], bins=binrule, density=True)
    p_lon, be_lon = np.histogram(link_points[:, 1], bins=binrule, density=True)
    p, b_lat, b_lon = np.histogram2d(link_points[:,0], link_points[:,1],
                                     bins=[be_lat, be_lon], density=True)
    
    p_dataarray = xr.DataArray(data=p, coords={'lat': b_lat[:-1], 'lon': b_lon[:-1]}, name='density')
    Z = p_dataarray.interp({'lon': coord_rad[:,1], 'lat': coord_rad[:, 0]}, method='linear').data
    
    return Z


def compute_stats(runs):
    mean = np.mean(runs, axis = 0)
    std = np.std(runs, axis = 0)
    perc90 = np.quantile(runs, 0.9, axis = 0)
    perc95 = np.quantile(runs, 0.95, axis = 0)
    perc99 = np.quantile(runs, 0.99, axis = 0)
    perc995 = np.quantile(runs, 0.995, axis = 0)
    perc999 = np.quantile(runs, 0.999, axis = 0)

    return np.array([mean, std, perc90, perc95, perc99, perc995, perc999])

def link_bundle_null_model_link_number(coord_rad, num_links,
                                       folder, filename, bw,
                                       num_rand_permutations=1000):
    """
    Args:
    -----
    coord_rad: np.ndarray (num_nodes, 2)

    num_link: int

    folder: str

    filename: str

    bw: float

    num_rand_permutations: int

    """
    if num_links < 1:
        print("No number of links!")
        return None

    filename += f'_num_links_{num_links}.npy'

#     if os.path.exists(folder+filename):
#        return None

    null_model_bundles = np.zeros((num_rand_permutations, coord_rad.shape[0]))

    for s in range(num_rand_permutations):
        all_links_rand = np.vstack([np.random.choice(coord_rad[:,0], num_links),
                                    np.random.choice(coord_rad[:,1], num_links)]).T
        null_model_bundles[s,:] = spherical_kde(all_links_rand, coord_rad, bw)

    stats = compute_stats(runs=null_model_bundles)
    np.save(folder + filename, stats)

    return None
    

def link_bundle_null_model(adj_matrix, coord_rad,
                           link_bundle_folder, filename, bw, num_rand_permutations=2000,
                           num_cpus=mpi.cpu_count()):
    """
    Args:
    -----
    adj_matrix: np.ndarray (num_nodes, num_nodes)
        Adjacency matrix
    coord_rad: np.ndarray (num_nodes, 2)
        Map coordinates of nodes in rad [lat, lon]
    link_bundle_folder: str

    bw: float
        KDE Bandwidth

    filename: str

    num_rand_permutations: int

    """

    buff = []
    for i in range(0, adj_matrix.shape[0]):
        count_row = np.count_nonzero(adj_matrix[i,:])
        count_col = np.count_nonzero(adj_matrix[:,i])
        buff.append(count_row)
        buff.append(count_col)

    # shuffle link numbers to have fairer job times 
    with temp_seed():  
        link_numbers = np.unique(buff)
        np.random.shuffle(link_numbers)

    if not os.path.exists(link_bundle_folder):
        os.makedirs(link_bundle_folder)
        print(f"Created folder: {link_bundle_folder}!")
    else:
        print(f"Save to folder: {link_bundle_folder}!")
    
    #for job array on slurm cluster
    try:
        min_job_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
        max_job_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        num_jobs = max_job_id
        print(f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}" )

    except KeyError:
        job_id = 0
        num_jobs = 1
        print("Not running with SLURM job arrays, but with manual id: ", job_id)

    diff_links = len(link_numbers)
    one_array_length = int(diff_links/num_jobs) + 1

    start_arr_idx = job_id * one_array_length
    end_arr_idx = (job_id + 1) * one_array_length

    # For parallel Programming
    print(f"Number of available CPUs: {num_cpus} for link bundeling!")
    print(f"Number of different number of links {len(link_numbers)}.")
    backend = 'multiprocessing'
    (Parallel(n_jobs=num_cpus, backend=backend)
             (delayed(link_bundle_null_model_link_number)
              (coord_rad, num_links,
               link_bundle_folder, filename, bw, num_rand_permutations)
              for num_links in tqdm(link_numbers[start_arr_idx:end_arr_idx]))
     )

    return None


def link_bundle_one_location(adj_matrix, idx_node, coord_rad,
                             folder, filename, bw,
                             perc=999, plot=False):
    """
    Args:
    -----
    adj_matrix: np.ndarray (num_nodes, num_nodes)

    coord_rad: np.ndarray (num_nodes, 2)

    Return:
    -------
    Dictionary with significant links

    """
    result_dic = {'idx_node': idx_node,
                  'significant_links': [],
                  'density': []}
    # Get links of node
    link_indices_node = np.where(adj_matrix[idx_node, :] > 0)[0]
    num_links = len(link_indices_node)
    link_coord = coord_rad[link_indices_node]
    if num_links < 1:
        print(f'Node with index {idx_node} has no links!')
        return result_dic

    # KDE
    Z_node = spherical_kde(link_coord, coord_rad, bw)

    # read null model
    filename += f'_num_links_{num_links}.npy'
    if not os.path.exists(folder+filename):
        raise ValueError(f"Warning {folder}/{filename} does not exist, even though #links={num_links}>1!")

    mean, std, perc90, perc95, perc99, perc995, perc999 = np.load(folder + filename)

    if perc == 999:
        Z_rand = perc999
    elif perc == 995:
        Z_rand = perc995
    elif perc == 99:
        Z_rand = perc99
    elif perc == 95:
        Z_rand = perc95
    elif perc == 90:
        Z_rand = perc90
    else:
        raise ValueError("Choosen percintile does not exist!")
    
    # Check if density is significant
    significant_indices = np.intersect1d(
        np.where(Z_node > Z_rand)[0], link_indices_node
    )
    result_dic['significant_links'] = significant_indices

    if plot:
        sigdat = Z_node.copy()
        sigdat[Z_node > mean + 5 * std] = 5.5
        sigdat[Z_node <= mean + 5 * std] = 4.5
        sigdat[Z_node <= mean + 4 * std] = 3.5
        sigdat[Z_node <= mean + 3 * std] = 2.5
        sigdat[Z_node <= mean + 2 * std] = 1.5
        sigdat[-1] = 1.5
        sigdat[0] = 1.5

        result_dic['density'] = sigdat
    else:
        result_dic['density'] = Z_node

    return result_dic


def link_bundle_adj_matrix(adj_matrix, coord_rad,
                           null_model_folder, null_model_filename, 
                           bw, perc=999,
                           num_cpus=mpi.cpu_count()):
    """
    """
    print(f"Number of available CPUs: {num_cpus}")

    backend='multiprocessing'
    results = (
        Parallel(n_jobs=num_cpus, backend=backend)
        (delayed( link_bundle_one_location)
            (adj_matrix, idx_node, coord_rad,
             null_model_folder, null_model_filename, bw, perc)
            for idx_node in tqdm(range(0, adj_matrix.shape[0])) 
        )
    )

    # Now update Adjacency Matrix
    adj_matrix_corrected = np.zeros_like(adj_matrix)
    for result_dic in results:
        idx_node = result_dic['idx_node']
        links = result_dic['significant_links']

        adj_matrix_corrected[idx_node, links] = 1
        
    return adj_matrix_corrected