import networkx as nx
import os, pickle
import numpy as np
import scipy.interpolate as interp
from scipy import special,integrate,stats
from tueplots import fontsizes
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as ctp
import networkx.algorithms as algs
from climnet.similarity_measures import *
import time
import pandas as pd
from scipy.stats import norm, rankdata
from climnet.grid import haversine
import fnmatch
from tueplots import bundles

from sklearn.covariance import OAS, LedoitWolf
from sklearn.metrics import mutual_info_score
from climnet.event_synchronization import event_synchronization_matrix

from matplotlib.colors import LinearSegmentedColormap#from numpy.lib.type_check import nan_to_num
import scipy

from contextlib import contextmanager
import tqdm, sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

@np.vectorize
def MP_density(x,c):
    upper = (1+np.sqrt(c)) ** 2
    lower = (1-np.sqrt(c)) ** 2
    if (x < lower) or (x > upper):
        return 0
    return np.sqrt((upper-x)*(x-lower)) / (2*np.pi * c * x)

def rank_matrix(mat):
    # ranks the entries != 0 (e.g. only the links)
    mat[mat != 0] = rankdata(np.abs(mat[mat != 0]))
    return mat


def W2_normal(mu1,mu2, sigma1,sigma2):
    w2squared = np.abs(mu1-mu2) ** 2 + np.trace(sigma1+sigma2-2 * scipy.linalg.sqrtm(sigma1 @ sigma2))
    return np.sqrt(w2squared)

def autocorr(X, t = 1):
    '''
    Computes autocorr at lag 1 between columns of X.
    '''
    d = X.shape[1]
    return np.corrcoef(X[:-t,:], X[t:,:],rowvar = False)[:d,d:]

def binstoplot(bins):
    return np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

def near_psd(A, epsilon = 1e-8):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = epsilon
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def upper(mat):
    upper = np.triu(np.ones_like(mat), k = 1)
    return mat[upper != 0]

def upper_to_matrix(uparray, diag = 1):
    dim = int(0.5 + np.sqrt(0.25 + 2 * len(uparray)))
    mat = np.zeros((dim,dim))
    upper = np.triu(np.ones_like(mat), k = 1)
    mat[upper != 0] = uparray
    mat = mat.T
    mat[upper != 0] = uparray
    mat[np.eye(dim, dtype = bool)] = diag
    return mat

def ij_from_idx(idx,m,n=None):
    if n is None:
        n = m
    # idx = i * n + j
    j = idx % n
    i = int((idx - j)/n)
    return i,j

def quant(alpha, l):
    return int(np.ceil(alpha*l-1))

def quantile(arr, alpha = 0.95):
    return np.sort(arr)[int(np.ceil(alpha*len(arr)-1))]

def get_adj(corrs, dens, weighted = True):
    corrs = np.abs(corrs - np.diag(np.diag(corrs)))
    corrvals = upper(corrs)
    thres = np.sort(corrvals)[quant(1-dens,len(corrvals))]
    adj = np.where(corrs >= thres, 1, 0)
    adj = adj - np.diag(np.diag(adj))
    if weighted:
        adj = adj * corrs
    return adj

def knn_adj(abscorr, k, directed = False, weighted = True):
    diag = np.eye(len(abscorr), dtype = bool)
    abscorr[diag] = -np.inf
    nn_idx = np.argsort(abscorr, axis=1)[:,-k:]
    adj = np.zeros_like(abscorr)
    for j in range(len(adj)):
        adj[j,nn_idx[j,:]] = 1 # still directed: j is k-nn to i
    if not directed:
        adj = np.maximum(adj.T, adj)
    abscorr[diag] = 0
    if weighted:
        adj = adj * abscorr
    return adj



def get_significance_adj(data, alpha = 0.95, num_shuffles = 1000, weighted = True, use_iaaft = False):
    adj = np.zeros((data.shape[1],data.shape[1]))
    corr = np.corrcoef(data.T)
    for i in range(data.shape[1]):
        for j in range(i):
            eps =  data[:,[i,j]]
            shufflecorrs = []
            if use_iaaft:
                iaaft = np.zeros((2, num_shuffles, data.shape[0]))
                with suppress_stdout():
                    for ipt in range(eps.shape[1]):
                        iaaft[ipt,:,:] = surrogates(eps[:,ipt], ns = num_shuffles)
                for isurr in range(num_shuffles):
                    shufflecorrs.append(np.corrcoef(iaaft[:,isurr,:])[0,1])
            else:
                for _ in range(num_shuffles):
                    eps[:,0] = eps[np.random.permutation(data.shape[0]),0]
                    shufflecorrs.append(np.corrcoef(eps.T)[0,1])
            if corr[i,j] >= quantile(shufflecorrs,alpha = alpha):
                adj[i,j] = 1
                adj[j,i] = 1
    if weighted:
        adj = corr * adj
    return adj

def compute_empcorr(data, similarity = 'pearson', q_mi = 2, unbiased_HSIC = True, permute_HSIC = False, alpha_es = 0.05):
    emp_corr = np.zeros((data.shape[1], data.shape[1]))
    if similarity == 'pearson':
        return np.corrcoef(data.T)
    elif similarity == 'spearman':
        emp_corr, pval = stats.spearmanr(data)
    elif similarity == 'LW':
        emp_corr = lw_analytical_shrinkage_estimator(data)[0]
    elif similarity == 'LWlin':
        lw = LedoitWolf()
        emp_corr = lw.fit(data).covariance_
    elif similarity == 'OAS':
        oa = OAS()
        emp_corr = oa.fit(data).covariance_
    elif similarity == 'BI-KSG':
        for i in range(data.shape[1]):
            for j in range(i):
                emp_corr[i,j] = revised_mi(data[:,i], data[:,j], q = q_mi)
                emp_corr[j,i] = emp_corr[i,j]
    elif similarity == 'binMI':
        emp_corr = calculate_mutual_information(data)
    elif similarity == 'HSIC':
        for i in range(data.shape[1]):
            for j in range(i):
                hsic = HSIC(data[:,i], data[:,j],unbiased=unbiased_HSIC, permute = permute_HSIC)
                emp_corr[i,j] = hsic
                emp_corr[j,i] = emp_corr[i,j]
    elif similarity == 'ES':
        sorted_dat = np.sort(np.abs(data),axis=0)
        dat_quantiles = sorted_dat[int(np.ceil(alpha_es * len(sorted_dat)-1)),:]
        events= (np.abs(data).T >= np.repeat(dat_quantiles.reshape((-1,1)),data.shape[0], axis = 1))
        emp_corr = event_synchronization_matrix(events)
    else:
        raise RuntimeError('Unknown similarity measure.')
    return emp_corr

def get_bootcorr(data, seed,corr_method = 'pearson'):
    np.random.seed(seed)
    boot_idcs = np.random.choice(data.shape[0], data.shape[0], replace = True)
    return compute_empcorr(data[boot_idcs,:],similarity=corr_method)

def compute_plink_distance(A,n_time, similarity = 'pearson', distrib = 'igrf', num_iter = 1000, n_resol = 100, n_dresol = 100):
    dis = (np.arange(n_dresol)+0.5) * np.pi / n_dresol
    # thres
    xs = np.arange(n_resol) / n_resol
    corrs = np.zeros((n_dresol,num_iter))
    for i,d in enumerate(dis):
        #corrs = []
        for it in range(num_iter):
            if distrib == 'igrf':
                data = (np.random.multivariate_normal(np.zeros(2), cov = var * np.array([[1, corr_igrf_from_dist(d,A)],[corr_igrf_from_dist(d,A), 1]]), size= n_time))
            elif distrib == 'expigrf':
                data = np.exp(np.random.multivariate_normal(np.zeros(2), cov = var * np.array([[1, corr_igrf_from_dist(d,A)],[corr_igrf_from_dist(d,A), 1]]), size= n_time))
                data[:,0] -= data[:,0].mean()
                data[:,0] /= data[:,0].std()
                data[:,1] -= data[:,1].mean()
                data[:,1] /= data[:,1].std()
            if similarity == 'pearson':
                corrs[i,it] = np.corrcoef(data.T)[0,1]
            elif similarity == 'spearman':
                corrs[i,it] = stats.spearmanr(data)[0]
            elif similarity == 'MI':
                corrs[i,it] = revised_mi(data[:,0], data[:,1], q = 2)
            elif similarity == 'HSIC':
                hsic[i,it] = HSIC(data[:,0], data[:,1],unbiased=True, permute = False)
    if similarity == 'MI' or similarity == 'HSIC':
        corrs /= np.sort(corrs, axis = None)[-num_iter]
    corrprobgeq = cnprobgeq(corrs,xs)
    return corrprobgeq

def get_mips(adj, n_estim = None):
    G = nx.from_numpy_matrix(adj)
    if n_estim is not None:
        idcs = np.random.permutation(adj.shape[0])[:n_estim]
        cc = np.array(list(nx.clustering(G, nodes = idcs).values()))
    else:
        cc = np.array(list(nx.clustering(G).values()))
    mip_cc = np.argsort(cc)[-10:] # most important points
    eigc = nx.eigenvector_centrality_numpy(G)
    eigc = np.array(list(eigc.values()))
    mip_eigc = np.argsort(eigc)[-10:]
    betw = nx.betweenness_centrality(G, k = n_estim)
    betw = list(betw.values())
    mip_betw = np.argsort(betw)[-10:]
    return mip_eigc, mip_cc, mip_betw

def get_densidx(realdens, denslist = np.logspace(-3,np.log(0.25)/np.log(10), num = 20)):
    '''
    Returns the indices in realdens closest to the entries in denslist.
    '''
    diff = lambda x: np.abs(realdens - x)
    idcs = np.zeros(len(denslist))
    for i, de in enumerate(denslist):
        idcs[i] = diff(de).argmin()
    return idcs.astype(int)

def gdistance(pt1, pt2, radius=6371.009):
    '''
    Computes haversine distance.
    '''
    lon1, lat1 = pt1[1], pt1[0]
    lon2, lat2 = pt2[1], pt2[0]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

def fdr(lls, true_lls):
    fdr = np.zeros((lls.shape[0],lls.shape[2]))
    for lineidx in range(lls.shape[0]):
        binidx = np.where(true_lls[lineidx,:]>0)[0][-1]
        fdr[lineidx,:] = lls[lineidx,(binidx+1):,:].sum(axis=0) / lls[lineidx,:,:].sum(axis=0)
    return fdr

def calc_MI(x, y, bins):
    '''
    Computes binned MI estimator. (Bad idea)
    '''
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def mysave(path,name,data):
    if os.path.exists(path):
        if os.path.exists(path+name):
            os.remove(path+name)
        with open(path+name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
    else:
        os.mkdir(path)
        with open(path + name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)

def myload(name):
    with open(name, "rb") as fp:   # Unpickling
        all_stats = pickle.load(fp)
    return all_stats

def find(pattern, path, nosub = False):
    '''
    Returns list of filenames containing pattern in path.
    '''
    result = []
    if nosub:
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for name in all_files:
            if fnmatch.fnmatch(name,pattern):
                result.append(os.path.join(path, name))
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
    return result

def standardize(dataset, axis=0):
    return (dataset - np.average(dataset, axis=axis)) / (np.std(dataset, axis=axis))


def get_bootdata(data, block_size = 1, seed = None):
    n, d = data.shape
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    block_idcs = np.random.choice(n // block_size, n // block_size)
    bootdata = np.zeros((block_size * (n // block_size),d))
    for i,blockidx in enumerate(block_idcs):
        bootdata[int(i * block_size):int((i+1)*block_size),:] = data[int(blockidx * block_size):int((blockidx+1)*block_size),:]
    return bootdata

# compute distances on the grid in lon, lat
def dist_from_coords(lats, longs):
    num_longs = len(longs)
    num_lats = len(lats)
    out = -1 * np.ones((num_longs, num_lats, num_longs, num_lats))
    for i in range(len(longs)):
        for j in range(len(lats)):
            for k in range(num_longs):
                for l in range(num_lats):
                    out[i,j,k,l] = gdistance((lats[j], longs[i]), (lats[l],longs[k]))
    return out

def get_ptidx(lon, lat, grid):
    # computes the idx of the point in grid closest to (lon,lat)
    diff = np.concatenate(((grid['lon']-lon).reshape((-1,1)), (grid['lat']-lat).reshape((-1,1))),axis = 1)
    return np.linalg.norm(diff,axis = 1).argmin()

def coords_to_idx(i, j, num_longs):
    return i * num_longs + j

def idx_to_coords(idx, num_longs):
    i = idx // num_longs
    return (i, idx - i * num_longs)

def dist_from_idx(i,j, lats, longs):
    num_longs = len(longs)
    coords0 = idx_to_coords(i, num_longs)
    coords1 = idx_to_coords(j, num_longs)
    return gdistance((lats[coords0[0]], longs[coords0[1]]), (lats[coords1[0]],longs[coords1[1]]))

def all_dists(lats, longs, regular = True): # calc all dists in idx format
    if regular:
        num_longs = len(longs)
        n_points = num_longs * len(lats)
        dists = -1 * np.ones((n_points, n_points))
        for i in range(n_points):
            dists[i,i] = 0
            for j in range(i):
                coords0 = idx_to_coords(i, num_longs)
                coords1 = idx_to_coords(j, num_longs)
                dists[i,j] = gdistance((lats[coords0[0]], longs[coords0[1]]), (lats[coords1[0]],longs[coords1[1]]))
                dists[j,i] = dists[i,j]
    else:
        dists = -1 * np.ones((len(longs), len(longs)))
        for i in range(len(longs)):
            dists[i,i] = 0
            for j in range(i):
                dists[i,j] = gdistance((lats[i], longs[i]), (lats[j],longs[j]))
                dists[j,i] = dists[i,j]
    return dists

#for each edge in g append its distance
def edge_distances(g,longs,lats):
    edge_dists = []
    for e in g.edges:
        edge_dists.append(dist_from_idx(e[0],e[1],longs,lats))
    return edge_dists

# returns a dict: degree: number of nodes
def get_degrees(g):
    degree_sequence = sorted([d for n, d in g.degree()]) #, reverse=True
    degs = {}
    for deg in degree_sequence:
        if deg not in degs.keys():
            degs[deg] = 1
        else:
            degs[deg] += 1
    return degs

def gridstep_to_numpoints(gridstep):
    dist_equator = gdistance((0,0),(0,gridstep))
    k = -2.01155176
    a = np.exp(20.0165958)
    return int(a * dist_equator**k)

def meshplot(x,y,z, title, minn = None, maxx = None):
        if minn is None:
                minn = z.min()
        if maxx is None:
                maxx = z.max()
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=minn, vmax=maxx)
        ax.set_title(title)
        ax.set_xlabel('Threshold')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
        return fig, ax

def mean_weight(subadj):
    '''
    Mean weight in flattened subadj, where rows and cols represent same nodes.
    '''
    l = np.sqrt(len(subadj))
    return subadj.sum()/(l*(l-1))

def decorr_length(adj,dists, min_connectivity = 0.5, grid_type = None, base_path = None):
    if grid_type is None:
        distidcs = np.argsort(dists,axis=1)
    else:
        if os.path.exists(base_path+f'distidcs_{grid_type}{dists.shape[0]}.txt'):
            distidcs = myload(base_path+f'distidcs_{grid_type}{dists.shape[0]}.txt')
        else:
            distidcs = np.argsort(dists)
            mysave(base_path, f'distidcs_{grid_type}{dists.shape[0]}.txt', distidcs)           
    sdists = np.sort(dists,axis = 1)
    sadj = np.take_along_axis(adj,distidcs,axis=1)#np.sort(adj,axis = 1)
    #print(sdists[0,:],sadj[0,:])
    lengths = np.zeros(len(adj))
    # for i in range(len(adj)-1):
    #     sadj[:,i+1] += sadj[:,i]
    # for i in range(len(adj)):
    #     sadj[:,i] /= i      
    for i in range(len(adj)): # for each point
        try:
            radius_idx = np.where(np.array([sadj[i,1:(idx+2)].mean() for idx in range(len(sadj)-2)]) >= min_connectivity)[0][-1]
            #print(sdists[i,:],np.where(np.array([sadj[i,1:(idx+2)].mean() for idx in range(len(sadj))]) >= min_connectivity)[0])
            lengths[i] = sdists[i,radius_idx+1]
        except IndexError:
            lengths[i] = 0
    
    return lengths

def bundlefraction(adj, dists, nbhd, maxdist, tolerance = 0.1, typ = 'lw'): # 'raw', '1tm'
    # false edges in bundles / false edges
    # assumes 2 eps <= maxdist
    longlinks = np.where(np.logical_and(adj != 0, dists > maxdist))
    if len(longlinks[0]) == 0:
        return 0
    bundlecount = 0
    for ilink in range(len(longlinks[0])):
        i,j = longlinks[0][ilink], longlinks[1][ilink]
        if i > j:
            continue
        nbhdi = nbhd[i]
        nbhdj = nbhd[j]
        if typ == 'raw':
            subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
            if subadj.sum()/ (len(nbhdi) * len(nbhdj)) >= 1-tolerance:
                bundlecount += 1
        elif typ == 'lw':
            subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
            subi = adj[np.repeat(nbhdi,len(nbhdi)), np.tile(nbhdi, len(nbhdi))]
            subj = adj[np.repeat(nbhdj,len(nbhdj)), np.tile(nbhdj, len(nbhdj))]
            if subadj.sum() / (len(nbhdi) * len(nbhdj)) >= (1-tolerance) * (mean_weight(subi) / 2 + mean_weight(subj) / 2):
                bundlecount += 1
        elif typ == '1tm':
            if (adj[np.repeat(i,len(nbhdj)), nbhdj].sum() / len(nbhdj) >= 1-tolerance) or (adj[np.repeat(j,len(nbhdi)), nbhdi].sum() / len(nbhdi) >= 1-tolerance):
                bundlecount += 1
        else:
            raise ValueError('typ not in lw, raw, 1tm.')
    return bundlecount / len(longlinks[0])


def bundlefractionwhere(adj, diff, nbhd, tolerance = 0.1, typ = 'lw'): # 'raw', '1tm'
    # false edges in bundles / false edges
    # assumes 2 eps <= maxdist
    longlinks = diff
    if len(longlinks[0]) == 0:
        return 0
    bundlecount = 0
    for ilink in range(len(longlinks[0])):
        i,j = longlinks[0][ilink], longlinks[1][ilink]
        if i > j:
            continue
        nbhdi = nbhd[i]
        nbhdj = nbhd[j]
        if typ == 'raw':
            subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
            if subadj.sum()/ (len(nbhdi) * len(nbhdj)) >= 1-tolerance:
                bundlecount += 1
        elif typ == 'lw':
            subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
            subi = adj[np.repeat(nbhdi,len(nbhdi)), np.tile(nbhdi, len(nbhdi))]
            subj = adj[np.repeat(nbhdj,len(nbhdj)), np.tile(nbhdj, len(nbhdj))]
            if subadj.sum() / (len(nbhdi) * len(nbhdj)) >= (1-tolerance) * (mean_weight(subi) / 2 + mean_weight(subj) / 2):
                bundlecount += 1
        elif typ == '1tm':
            if (adj[np.repeat(i,len(nbhdj)), nbhdj].sum() / len(nbhdj) >= 1-tolerance) or (adj[np.repeat(j,len(nbhdi)), nbhdi].sum() / len(nbhdi) >= 1-tolerance):
                bundlecount += 1
        else:
            raise ValueError('typ not in lw, raw, 1tm.')
    return bundlecount / len(longlinks[0])

#calc furthest eps-stable teleconnection
def teleconn(adj, dists, eps, tolerance = 0.1):
    d = 0
    complete_adj = np.ones_like(adj)
    complete_adj[np.eye(adj.shape[0],dtype=bool)] = 0
    for i in range(len(adj)):
        for j in range(i):
            if adj[i,j] != 0 and dists[i,j] > d:
                # if all points close to j are connected to i, update d
                #  points close to j # connected to i
                nbhdj = np.where(dists[j,:] <= eps)[0] #, np.where(adj[i,:] != 0))
                subadj = adj[np.repeat(i,len(nbhdj)), nbhdj]
                subcomplete = complete_adj[np.repeat(i,len(nbhdj)), nbhdj]
                if subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                    d = dists[i,j]
                #if np.all(np.isin(nbhdj, np.where(adj[i,:] != 0))):
                #    d = dists[i,j]
    return d

#calc furthest robust eps-stable teleconnection
def robust_teleconn(adj, dists, eps, tolerance = 0.1, raw = False):
    '''
    Raw requires avg. between-weight to be above 1-tolerance, else: avg. between-weight at least (1-tolerance) of avg. within-weight
    '''
    d = 0
    complete_adj = np.ones_like(adj)
    complete_adj[np.eye(adj.shape[0],dtype=bool)] = 0
    for i in range(len(adj)):
        nbhdi = np.where(dists[i,:] <= eps)[0]
        subi = adj[np.repeat(nbhdi,len(nbhdi)), np.tile(nbhdi, len(nbhdi))]
        for j in range(i):
            if adj[i,j] != 0 and dists[i,j] > d:
                # if all points close to j are connected to i, update d
                #  points close to j # connected to i
                nbhdj = np.where(dists[j,:] <= eps)[0] #, np.where(adj[i,:] != 0))
                subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                subcomplete = complete_adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                if raw:
                    if subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
                else:
                    subj = adj[np.repeat(nbhdj,len(nbhdj)), np.tile(nbhdj, len(nbhdj))]
                    if subadj.sum() / subcomplete.sum() >= (1-tolerance) * (mean_weight(subi) / 2 + mean_weight(subj) / 2):  #subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
    return d

def robust_teleconn_nbhd(adj,dists, nbhd, tolerance = 0.1, raw = False):
    '''
    Uses precalculated neighborhoods of points and checks whether link density between these is above 1-tolerance. 
    Raw requires avg. between-weight to be above 1-tolerance, else: avg. between-weight at least (1-tolerance) of avg. within-weight
    '''
    d = 0
    complete_adj = np.ones_like(adj)
    complete_adj[np.eye(adj.shape[0],dtype=bool)] = 0
    for i in range(len(adj)):
        nbhdi = nbhd[i]
        subi = adj[np.repeat(nbhdi,len(nbhdi)), np.tile(nbhdi, len(nbhdi))]
        for j in range(i):
            if adj[i,j] != 0 and dists[i,j] > d:
                # if all points close to j are connected to i, update d
                #  points close to j # connected to i
                nbhdj = nbhd[j] #, np.where(adj[i,:] != 0))
                subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                subcomplete = complete_adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                if raw:
                    if subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
                else:
                    subj = adj[np.repeat(nbhdj,len(nbhdj)), np.tile(nbhdj, len(nbhdj))]
                    if subadj.sum() / subcomplete.sum() >= (1-tolerance) * (mean_weight(subi) / 2 + mean_weight(subj) / 2):  #subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
    return d

def maxavgdeg(adj, nbhds):
    # compute maximal avg degree in eps-nhbds
    l = len(adj)
    max_deg = 0
    for i in range(l):
        nbhdi = nbhds[i]
        avgdegi = adj[nbhdi, :].sum()/((l-1) * len(nbhdi))
        if avgdegi > max_deg:
            max_deg = avgdegi
    return max_deg

def maxavgdegbias(adj,nbhds, num_perm = 100):
    l = len(adj)
    perm_maxavgdeg = np.zeros(num_perm)
    for i in range(num_perm):
        perm_idcs = np.random.permutation(l)
        perm_maxavgdeg[i] = maxavgdeg(adj[perm_idcs,:][:,perm_idcs],nbhds) 
    return maxavgdeg(adj,nbhds), perm_maxavgdeg


# calc furthest robust teleconnection if large value expected
def robust_teleconn_dense(adj,dists, eps, tolerance = 0.1, raw = False):
    d = 0
    complete_adj = np.ones_like(adj)
    complete_adj[np.eye(adj.shape[0],dtype=bool)] = 0
    sortidcs = np.argsort(dists,axis=1)[:,::-1]
    # go from furthest to closest
    for furth_i in range(len(adj)):
        for j in range(len(adj)):
            #i is furth_i-furthest point from j
            i = sortidcs[j,furth_i]
            if adj[i,j] != 0 and dists[i,j] > d:
                # if most points close to j are connected to most points close to i, update d
                nbhdi = np.where(dists[i,:] <= eps)[0]
                nbhdj = np.where(dists[j,:] <= eps)[0] #, np.where(adj[i,:] != 0))
                subadj = adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                subcomplete = complete_adj[np.repeat(nbhdi,len(nbhdj)), np.tile(nbhdj, len(nbhdi))]
                if raw:
                    if subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
                else:
                    subi = adj[np.repeat(nbhdi,len(nbhdi)), np.tile(nbhdi, len(nbhdi))]
                    subj = adj[np.repeat(nbhdj,len(nbhdj)), np.tile(nbhdj, len(nbhdj))]
                    if subadj.sum() / subcomplete.sum() >= (1-tolerance) * (mean_weight(subi) / 2 + mean_weight(subj) / 2):  #subadj.sum()/ subcomplete.sum() >= 1-tolerance:
                        d = dists[i,j]
        if d > 0:
            return d
    return d

def lldistr(dists,adj, bins):
    lld = np.zeros((len(bins)-1))
    for i in range(len(bins)-1):
        l,u = bins[i], bins[i+1]
        lld[i] = adj[np.logical_and(dists>=l, dists < u)].sum()
    return lld / 2

def graph_measures(g):
    bc = algs.betweenness_centrality(g)
    katz= algs.katz_centrality(g)
    deg_centr = algs.degree_centrality(g)
    eig_centr = algs.eigenvector_centrality(g)
    cc = algs.closeness_centrality(g)
    deg_corr = nx.degree_pearson_correlation_coefficient(g)
    print("BC: ", bc)
    print("Katz: ", katz)
    print("Degree centr: ", deg_centr)
    print("Eigenvector centr: ", eig_centr)
    # algs.group_...(g,node_list)
    print("CC: ", cc)
    print("degree pearson correlation coeff: ", deg_corr)
    return(bc, katz, deg_centr, eig_centr, cc, deg_corr )


def get_same_data_used(datas): # shape (time, number of ts)
    same_data = []
    for i in range(datas.shape[1]):
        for j in range(i):
            if np.array_equal(datas[:,i], datas[:,j]):
                same_data.append((j,i))
    return same_data

def get_binned_ll_dist(Net, num_bins = 100, bins = None): # very inefficient!
    ll_dist = Net.get_link_length_distribution()
    if bins is None:
        a100 = np.linspace(np.log(ll_dist.min())/np.log(10), np.log(ll_dist.max())/np.log(10), num_bins)
        bins = 10 ** a100
    return plt.hist(ll_dist, bins = bins)

def get_loglog_bins(data, num_bins = 100):
    a100 = np.linspace(np.log(data.min())/np.log(10), np.log(data.max())/np.log(10), num_bins)
    return 10 ** a100

def links_between(idx0, idx1, radius, dists, Net):
    close0 = np.where(dists[idx0,:] <= radius)[0]
    close1 = np.where(dists[idx1,:] <= radius)[0]

    numlinks = 0
    for clo in close0:
        numlinks += Net.adjacency[clo,close1].sum()
    return numlinks, numlinks /(len(close0) * len(close1))

def gaussian_kernel(x, sig = 1, trunc = 0):
    vals = np.exp(-np.abs(x) ** 2/(2 * sig))
    return np.where(vals <= trunc, 0, vals)

def legendre_matern(k,nu,alphasq=1):
    return 4 * np.pi / ((2 * k+1) * (alphasq + k ** 2) ** (nu+0.5) )

def meshplot(x,y,z, title, minn = None, maxx = None):
        if minn is None:
                minn = z.min()
        if maxx is None:
                maxx = z.max()
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=minn, vmax=maxx)
        ax.set_title(title)
        ax.set_xlabel('Threshold')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(c, ax=ax)
        return fig, ax

def cnprobgeq(corrs, xs):
    cnprobgeq = np.zeros((corrs.shape[0], len(xs)))
    corrs.sort(axis = 1)
    for j, thres in enumerate(xs):
        corrwhere = np.where(corrs >= thres)
        for i in range(corrs.shape[0]):
            if i not in np.unique(corrwhere[0]):
                cnprobgeq[i,j] = 0
            else:
                cnprobgeq[i,j] = 1 - corrwhere[1][corrwhere[0] == i][0] / corrs.shape[1]
    return cnprobgeq

def get_quants(arr,xs):
    sortedarr = np.sort(arr,axis = None)
    l = len(sortedarr)
    quants = np.zeros_like(xs)
    for i,x in enumerate(xs):
        quants[i] = sortedarr[quant(x, l)]
    return quants

def diag_var_process(ar_coeff, cov, n_time, n_pre = 100, eps = 1e-8):
    dim = cov.shape[0]
    data = np.zeros((n_time+n_pre,dim))
    sigma = near_psd(cov - np.diag(ar_coeff) @ cov @ np.diag(ar_coeff), epsilon=eps)
    print('2-norm difference between cov matrix of epsilon and ideal cov matrix of eps:', np.linalg.norm(sigma-(cov - np.diag(ar_coeff) @ cov @ np.diag(ar_coeff)),2))
    eps = np.random.multivariate_normal(np.zeros(dim), sigma, size=n_time+n_pre)
    data[0,:] = np.random.multivariate_normal(np.zeros(dim), cov)
    for i in range(n_pre+n_time-1):
        data[i+1,:] = ar_coeff * data[i,:] + eps[i+1,:]
    return data[n_pre:,:]

def true_cov_diag_var(ar_coeff, cov, i_max = 100, eps = 1e-8):
    print('Does not work sufficiently well.')
    return None
    sigma = near_psd(cov - np.diag(ar_coeff) @ cov @ np.diag(ar_coeff), epsilon=eps)
    out = sigma
    for i in range(i_max):
        out += np.einsum('j,jk,k->jk',ar_coeff ** i,sigma,ar_coeff ** i)
    return out


def eig_var_process(eigvecs, eigvals, n_time, ar_pc, pc_idx, n_pre = 100):
    dim = eigvecs.shape[0]
    data = np.zeros((n_time+n_pre,dim))
    eigA = np.zeros_like(eigvecs)
    eigA[pc_idx,pc_idx] = ar_pc
    A = eigvecs @ eigA @ eigvecs.T
    eigvals_eps = eigvals
    eigvals_eps[pc_idx] = eigvals[pc_idx] * (1-ar_pc ** 2)
    sigma = eigvecs @ np.diag(eigvals_eps) @ eigvecs.T
    eps = np.random.multivariate_normal(np.zeros(dim), sigma, size=n_time+n_pre)
    data[0,:] = np.random.multivariate_normal(np.zeros(dim), eigvecs @ np.diag(eigvals) @ eigvecs.T)
    for i in range(n_pre+n_time-1):
        data[i+1,:] = A @ data[i,:] + eps[i+1,:]
    return data[n_pre:,:]

def generate_AR(beta, cov, time_len = 100, prerun = 100, distr = 'normal', seed = None):
    """Creates AR(len(beta)) process with noise covariance cov
    
    Parameters:
    -----------
    beta: list
        consists of 1d np.arrays giving the lag weights of the AR process (each only depends on previous instance of the same)
    cov: np.array of dim 2
        covariance matrix of the noise
    prerun: int
        how many iterations of the process before inspection
    distr:
        function that takes (number of samples, cov)

    Returns:
    --------
    np.array of shape (time_len, cov.shape[0])  
    """
    
    if seed is not None:
        np.random.seed(seed)
    if type(beta) == list:
        r = len(beta)
        print('r = ',r)
    else:
        print("r = 1")
        r = 0
    if distr == 'normal':
        noise = np.random.multivariate_normal(np.zeros_like(cov[:,0]),cov, time_len+prerun+r)
    else:
        noise = distr(time_len+prerun+r,cov)

    X = - 9999 * np.ones((time_len, cov.shape[0]))
    if type(beta) == list:
        Xinit = -9999 * np.ones((prerun+r, cov.shape[0]))
        for i in range(prerun+r):
            temp = noise[i,:]
            for j in range(r):
                if j < i:
                    temp += beta[j] @ Xinit[i-j-1,:]
            Xinit[i,:] = temp
            
        X[:r,:] = Xinit[-r:,:]
        for i in range(r,time_len):
            temp = noise[prerun+r+i,:]
            for j in range(r):
                temp += beta[j] @ X[i-j-1,:]
            X[i,:] = temp
    else:
        Xinit = noise[0,:]
        for i in range(prerun):
            Xinit = beta * Xinit + noise[i+1,:]
        
        X[0,:] = Xinit
        for i in range(time_len-1):
            X[i+1,:] = beta * X[i,:] + noise[prerun+1+i,:]
    return X

def linear_bump(x0, radius):
    def bump(lo,la):
        if radius - gdistance((la,lo),(x0[1],x0[0])) <= 0:
            return 0
        return (radius - gdistance((la,lo),(x0[1],x0[0])))/radius
    return bump

# field possibly dependent in time for fixed points
def tele_field(telebump, telebump2, localfield, cleanidcs, signal = None, sigma = 1, corruption = 0):
    # telebump, noisebump shape npoints
    # localfield shape ntime, npoints
    # signal shape ntime
    if localfield.shape[1] != telebump.shape[0]:
        raise(ValueError('Number of points missspecified.'))

    ntime, npoints = localfield.shape
    
    if signal is None:
        # standard normal signal in each time step
        signal = np.repeat(np.random.randn(ntime).reshape(ntime,1), npoints, axis = 1)
    signal2 = np.repeat(np.random.randn(ntime).reshape(ntime,1), npoints, axis = 1)
    
    telebump = np.repeat(telebump.reshape((1,npoints)), ntime, axis = 0)
    telebump2 = np.repeat(telebump2.reshape((1,npoints)), ntime, axis = 0)
    noise = np.random.randn(localfield.shape)
    noise[:,cleanidcs] = 0
    grfandtele = sigma * (1-telebump) * ((1-telebump2) * localfield + telebump2 * signal2) + sigma * telebump * signal
    return grfandtele + corruption * noise

# field possibly dependent in time for fixed points
def tele_field_noise(telebump, noisebump, localfield, signal = None, sigma = 1):
    # telebump, noisebump shape npoints
    # localfield shape ntime, npoints
    # signal shape ntime
    if localfield.shape[1] != telebump.shape[0]:
        raise(ValueError('Number of points missspecified.'))

    ntime, npoints = localfield.shape
    
    if signal is None:
        # standard normal signal in each time step
        signal = np.repeat(np.random.randn(ntime).reshape(ntime,1), npoints, axis = 1)
    
    telebump = np.repeat(telebump.reshape((1,npoints)), ntime, axis = 0)
    noisebump = np.repeat(noisebump.reshape((1,npoints)), ntime, axis = 0)
    noise = np.random.randn(localfield.shape)
    return sigma * (1-telebump) * ((1-noisebump) * localfield + noisebump * noise) + sigma * telebump * signal

def get_legendre(K):
    def leg(lat):
        return special.lpmn(K,K, lat)[0]
    return leg

def igrf_var(A):
    var = 0
    for l in range(len(A)):
        var += A[l] * (2 * l + 1) / (4 * np.pi)
    return var

def cov_func_igrf(lat, lon, A, idx = 0):
    '''
    Computes true covariance values of igrf with spectrum A between (lon[idx], lat[idx]) and all other
    '''
    out = np.zeros_like(lat)
    cos_dist = np.cos(haversine(lon[idx], lat[idx], lon, lat))
    for l in range(len(A)):
        out += A[l] * (2 * l + 1) * special.legendre(l)(cos_dist) / (4 * np.pi)
    return out

def corr_func_igrf(lat, lon, A, idx = 0):
    '''
    Computes true correlation values of igrf with spectrum A between (lon[idx], lat[idx]) and all other
    '''
    var = 0
    for l in range(len(A)):
        var += A[l] * (2 * l + 1) / (4 * np.pi)
    cov = cov_func_igrf(lat,lon,A, idx)
    return cov / var
    
def corr_igrf_from_dist(d,A):
    return corr_func_igrf([0.0,0.0], [0.0,d * 180/np.pi], A)[1]

def normal_correlation(r, corr, n = 100):
    num = (n-2) * special.gamma(n-1) * (1-corr ** 2) **((n-1)/2) * (1- r ** 2) ** ((n-4)/2) * special.hyp2f1(0.5,0.5,0.5*(2*n-1), 0.5 * (corr*r+1))
    denom = np.sqrt(2* np.pi) * special.gamma(n-0.5) * (1- corr * r) ** (n-1.5)
    return num / denom 

def probgeq(corr, thres, n = 100):
    return integrate.quad(lambda x: normal_correlation(x,corr,n), thres,1) + integrate.quad(lambda x: normal_correlation(x,corr,n), -1, -thres)

def lognormal_corr_from_dist(d,A):
    var = igrf_var(A)
    corr = (np.exp(var*corr_igrf_from_dist(d,A))-1)/(np.exp(var)-1)
    return corr

def isotropic_grf(A, rvs = None):
    # A 1d array spectrum
    root_A = np.sqrt(A)
    K = len(A) - 1
    if rvs is None:
        rvs = np.random.normal(size = (K+1) ** 2 )
    # leg = get_legendre(K) 
    def T(lat,lon):
        theta = np.pi / 2 - np.pi * lat / 180
        phi = np.pi * lon / 180
        leg = special.lpmn(K,K, np.cos(theta))[0]
        out = 0
        for l in range(len(A)):
            out += root_A[l] * rvs[int(l ** 2)] * np.sqrt((2*l+1) / (4 * np.pi)) * leg[0,l]
            for m in range(1,l+1):
                if m == 1:
                    running_factorial = l * (l+1)
                else:
                    running_factorial *= (l+m) * (l-m+1)
                l_lm = np.sqrt((2*l+1) / (4 * np.pi * running_factorial) ) * leg[m,l]
                out += np.sqrt(2) * root_A[l] * l_lm * (rvs[int(l ** 2 + 2*m - 1)] * np.cos(m * phi) + rvs[int(l ** 2 + 2*m)] * np.sin(m * phi))
        return out
    return T

def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def plot_map_ds(ds, data, plot_type='scatter', central_longitude=0, central_latitude = 0,
            vmin=None, vmax=None, color='RdBu_r', bar=True,
            ax=None, ctp_projection="Mollweide", label= None, grid_step=2.5):
    """Simple map plotting using xArray.
    
    Parameters:
    -----------
    dmap: xarray.Dataarray
        Dataarray containing data, and coordinates lat, lon
    plot_type: str
        Plot type, currently supported "scatter" and "colormesh". Default: 'scatter'
    central_longitude: int
        Set central longitude of the plot. Default: 0
    vmin: float
        Lower limit for colorplot. Default: None
    vmax: float
        Upper limit for colorplot. Default: None
    color: str
        Colormap supported by matplotlib. Default: "RdBu"
    bar: bool
        If True colorbar is shown. Default: True
    ax: matplotlib.axes
        Axes object for plotting on. Default: None
    ctp_projection: str
        Cartopy projection type. Default: "Mollweide"
    label: str
        Label of the colorbar. Default: None
    grid_step: float
        Grid step for interpolation on Gaussian grid. Only required for plot_type='colormesh'. Default: 2.5

    Returns:
    --------
    Dictionary including {'ax', 'projection'}    
    """
    long_longs = ds.GridClass.grid['lon']
    long_lats = ds.GridClass.grid['lat']
    if ax is None:
        # set projection
        if ctp_projection == 'Mollweide':
            proj = ctp.crs.Mollweide(central_longitude=central_longitude)
        elif ctp_projection == 'PlateCarree':
            proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
        elif ctp_projection == "Orthographic":
            proj = ctp.crs.Orthographic(central_longitude, central_latitude)
        elif ctp_projection == "EqualEarth":
            proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
        else:
            raise ValueError(f'This projection {ctp_projection} is not available yet!')

        # create figure
        fig, ax = plt.subplots(figsize=(9,6))
        ax = plt.axes(projection=proj)
        ax.set_global()

        # axes properties
        ax.coastlines()
        ax.add_feature(ctp.feature.BORDERS, linestyle=':')
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    
        
    projection = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # set colormap
    cmap = plt.get_cmap(color)
    
    # plotting
    if plot_type =='scatter':
        im = ax.scatter(x=long_longs, y=long_lats,
                        c=data, vmin=vmin, vmax=vmax, cmap=cmap,
                        transform=projection)

    elif plot_type == 'colormesh':
        # interpolate grid of points to regular grid
        lon_interp = np.arange(-180,
                                180,
                                grid_step)
        lat_interp = np.arange(long_lats.min(),
                                long_lats.max() + grid_step,
                                grid_step)

        lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
        new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        origin_points = np.array([long_longs, long_lats]).T
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        new_values = interp.griddata(origin_points, data, new_points,
                                        method='nearest')
        mesh_values = new_values.reshape(len(lat_interp), len(lon_interp))

        im = ax.pcolormesh(
                lon_mesh, lat_mesh, mesh_values,
                cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)
    else:
        raise ValueError("Plot type does not exist!")

    if bar:
        label = ' ' if label is None else label
        cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                            label=label, shrink=0.8, ax=ax)

    return {"ax": ax,'fig': fig, "projection": projection}

def plot_deg_with_edges(grid, adj, transform = ctp.crs.PlateCarree(),label =None, ctp_projection='EqualEarth', color ='Reds', pts = None, *args):
    deg_plot = plot_map_lonlat(grid['lon'],grid['lat'], adj.sum(axis = 0), color=color,label =label, ctp_projection=ctp_projection, grid_step=grid_step_lon, *args)
    if pts is None:
        edges = np.where(adj != 0)
        for i in range(len(edges[0])):
            deg_plot['ax'].plot([grid['lon'][edges[0][i]],grid['lon'][edges[1][i]]], [grid['lat'][edges[0][i]], grid['lat'][edges[1][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
    else:
        for point in pts:
            edges = np.where(adj[point,:] != 0)
            for i in range(len(edges[0])):
                deg_plot['ax'].plot([grid['lon'][point], grid['lon'][edges[0][i]]], [grid['lat'][point], grid['lat'][edges[0][i]]], color = 'black', alpha = 0.3, linewidth = 0.5, transform=transform)
    return deg_plot

def plot_deg_with_edges_ds(grid, ds, adj, transform = ctp.crs.PlateCarree(), ctp_projection='EqualEarth', color ='Reds', pts = None, *args):
    deg_plot = plot_map_ds(ds, adj.sum(axis = 0), color=color, ctp_projection=ctp_projection, grid_step=ds.grid_step, *args)
    if pts is None:
        edges = np.where(adj != 0)
        for i in range(len(edges[0])):
            deg_plot['ax'].plot([grid['lon'][edges[0][i]],grid['lon'][edges[1][i]]], [grid['lat'][edges[0][i]], grid['lat'][edges[1][i]]], color = 'black', linestyle='--', transform=transform)
    else:
        for point in pts:
            edges = np.where(adj[point,:] != 0)
            for i in range(len(edges[0])):
                deg_plot['ax'].plot([grid['lon'][point], grid['lon'][edges[0][i]]], [grid['lat'][point], grid['lat'][edges[0][i]]], color = 'black', linestyle='--', transform=transform)
    return deg_plot


def plot_deg_with_edges_old(Net, transform = ctp.crs.PlateCarree(), ctp_projection='EqualEarth', color ='Reds', point = None, *args):
    grid = Net.dataset.grid
    deg_plot = plot_map_ds(Net.dataset, Net.adjacency.sum(axis = 0), color=color, ctp_projection=ctp_projection, grid_step=Net.dataset.grid_step, *args)
    if point is None:
        edges = np.where(Net.adjacency != 0)
        for i in range(len(edges[0])):
            deg_plot['ax'].plot([grid['lon'][edges[0][i]],grid['lon'][edges[1][i]]], [grid['lat'][edges[0][i]], grid['lat'][edges[1][i]]], color = 'black', linestyle='--', transform=transform)
    else:
        edges = np.where(Net.adjacency[point,:] != 0)
        for i in range(len(edges[0])):
            deg_plot['ax'].plot([grid['lon'][point], grid['lon'][edges[0][i]]], [grid['lat'][point], grid['lat'][edges[0][i]]], color = 'black', linestyle='--', transform=transform)
    return deg_plot

def plot_map_lonlat(lon, lat, data, plot_type='scatter', central_longitude=0, central_latitude = 0,
            vmin=None, vmax=None, color='RdBu_r', bar=True,cmap = None,
            ax=None, ctp_projection="Mollweide", label= None, grid_step=2.5, gridlines = True, earth = False,scale_const = 3, extend = 'both', norm = None,ticks=None):
    """Simple map plotting using xArray.
    
    Parameters:
    -----------
    dmap: xarray.Dataarray
        Dataarray containing data, and coordinates lat, lon
    plot_type: str
        Plot type, currently supported "scatter" and "colormesh". Default: 'scatter'
    central_longitude: int
        Set central longitude of the plot. Default: 0
    vmin: float
        Lower limit for colorplot. Default: None
    vmax: float
        Upper limit for colorplot. Default: None
    color: str
        Colormap supported by matplotlib. Default: "RdBu"
    bar: bool
        If True colorbar is shown. Default: True
    ax: matplotlib.axes
        Axes object for plotting on. Default: None
    ctp_projection: str
        Cartopy projection type. Default: "Mollweide"
    label: str
        Label of the colorbar. Default: None
    grid_step: float
        Grid step for interpolation on Gaussian grid. Only required for plot_type='colormesh'. Default: 2.5

    Returns:
    --------
    Dictionary including {'ax', 'projection'}    
    """
    long_longs = lon
    long_lats = lat
    
    # set projection
    if ctp_projection == 'Mollweide':
        proj = ctp.crs.Mollweide(central_longitude=central_longitude)
    elif ctp_projection == 'PlateCarree':
        proj = ctp.crs.PlateCarree(central_longitude=central_longitude)
    elif ctp_projection == "Orthographic":
        proj = ctp.crs.Orthographic(central_longitude, central_latitude)
    elif ctp_projection == "EqualEarth":
        proj = ctp.crs.EqualEarth(central_longitude=central_longitude)
    else:
        raise ValueError(f'This projection {ctp_projection} is not available yet!')

    if ax is None:
        # create figure
        fig, ax = plt.subplots(figsize=(scale_const*plt.rcParams['figure.figsize'][0],scale_const*plt.rcParams['figure.figsize'][1]))
    else:
        fig = None
    ax = plt.axes(projection=proj)
    ax.set_global()

    # axes properties
    if earth:
        ax.coastlines()
        ax.add_feature(ctp.feature.BORDERS, linestyle=':')
    if gridlines:
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    
        
    projection = ctp.crs.PlateCarree(central_longitude=central_longitude)

    # set colormap
    if cmap is None:
        cmap = plt.get_cmap(color)
    
    # plotting
    if plot_type =='scatter':
        if norm is None:
            im = ax.scatter(x=long_longs, y=long_lats,
                            c=data, vmin=vmin, vmax=vmax, cmap=cmap,
                            transform=projection)
        else:
            im = ax.scatter(x=long_longs, y=long_lats,
                            c=data, norm = norm, cmap=cmap,
                            transform=projection)
    elif plot_type == 'colormesh':
        # interpolate grid of points to regular grid
        lon_interp = np.arange(-180,
                                180,
                                grid_step)
        lat_interp = np.arange(long_lats.min(),
                                long_lats.max() + grid_step,
                                grid_step)

        lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
        new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        origin_points = np.array([long_longs, long_lats]).T
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        new_values = interp.griddata(origin_points, data, new_points,
                                        method='nearest')
        mesh_values = new_values.reshape(len(lat_interp), len(lon_interp))

        if norm is None:
            im = ax.pcolormesh(
                    lon_mesh, lat_mesh, mesh_values,
                    cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)
        else:
            im = ax.pcolormesh(
                    lon_mesh, lat_mesh, mesh_values,
                    cmap=cmap, norm=norm, transform=projection)
    else:
        raise ValueError("Plot type does not exist!")

    if bar:
        label = ' ' if label is None else label
        if ticks is None:
            cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                label=label, shrink=0.8, ax=ax)
        else:
            cbar = plt.colorbar(im, extend=extend, orientation='horizontal',
                                label=label, shrink=0.8, ax=ax,ticks=ticks)
            cbar.ax.set_xticklabels(ticks)
        cbar.set_label(label=label, size = scale_const*plt.rcParams['axes.labelsize'])
        cbar.ax.tick_params(labelsize = scale_const*plt.rcParams['xtick.labelsize'])
    return {"ax": ax,'fig': fig, "projection": projection}



def spherical2cartesian(lon, lat):
    lon = lon * 2 *np.pi / 360
    lat = lat * np.pi / 180
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return np.array([x, y, z]).T


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_fontsize(num_cols):
    keys = ['font.size','axes.labelsize','legend.fontsize','xtick.labelsize','ytick.labelsize','axes.titlesize']
    for key in keys:
        plt.rcParams[key] = bundles.icml2022()[key] * num_cols / 2
    


def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap







from tqdm import tqdm


def surrogates(x, ns, tol_pc=5., verbose=False, maxiter=1E6, sorttype="quicksort"):
    """
    Returns iAAFT surrogates of given time series.
    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.
    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.
    See Also
    --------
    numpy.argsort
    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    maxiter = 10000
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(ns), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs


# from ._ext.numerics import _randomly_rewire_geomodel_I, \
#         _randomly_rewire_geomodel_II, _randomly_rewire_geomodel_III

# pyunicorn.core.grid.Grid(time_seq, lat_seq, lon_seq, silence_level=0)

# def randomly_rewire_geomodel_II(adj, distance_matrix, iterations, inaccuracy):
#     """
#     Randomly rewire the current network in place using geographical
#     model II.

#     Geographical model II preserves the degree sequence :math:`k_v`
#     (exactly), the link distance distribution :math:`p(l)` (approximately),
#     and the average link distance sequence :math:`<l>_v` (approximately).

#     A higher ``inaccuracy`` in the conservation of :math:`p(l)` and
#     :math:`<l>_v` will lead to:

#         - less deterministic links in the network and, hence,
#         - more degrees of freedom for the random graph and
#         - a shorter runtime of the algorithm, since more pairs of nodes
#         eligible for rewiring can be found.

#     :type distance_matrix: 2D Numpy array [index, index]
#     :arg distance_matrix: Suitable distance matrix between nodes.

#     :type iterations: number (int)
#     :arg iterations: The number of rewirings to be performed.

#     :type inaccuracy: number (float)
#     :arg inaccuracy: The inaccuracy with which to conserve :math:`p(l)`.
#     """
    
#     #  Get number of nodes
#     N = distance_matrix.shape[0]
#     #  Get number of links
#     E = int(self.n_links)
#     #  Collect adjacency and distance matrices
#     A = self.adjacency.copy(order='c')
#     D = distance_matrix.astype("float32").copy(order='c')

#     #  Define for brevity
#     eps = float(inaccuracy)

#     #  Get edge list
#     edges = np.array(self.graph.get_edgelist()).copy(order='c')

#     _randomly_rewire_geomodel_II(iterations, eps, A, D, E, N, edges)

#     #  Update all other properties of GeoNetwork
#     self.adjacency = A



def geomodel2(adj, dists, eps, n_iter = 1000):
    edgeidcs = np.where(adj != 0)
    # want each edge only once
    num_edges2 = len(edgeidcs[0])
    deleteidcs = []
    for i in range(num_edges2):
        if edgeidcs[0][i] >= edgeidcs[1][i]:
            deleteidcs.append(i)
    fromidcs = np.delete(edgeidcs[0],deleteidcs)
    toidcs = np.delete(edgeidcs[1],deleteidcs)
    num_edges = len(fromidcs)
    edgeidcs = np.array([fromidcs,toidcs])

    it = 0
    count = 0
    while it < n_iter:
        # randomly choose two edges and then randomly choose two other indices in their range:
        ed1, ed2 = np.random.randint(num_edges,size=2)
        s,t = edgeidcs[0][ed1],edgeidcs[1][ed1]
        k,l = edgeidcs[0][ed2], edgeidcs[1][ed2]
        count += 1
        # four different nodes
        if s != k and s != l and t != k and t != l:
            # not yet connected
            if adj[s,l] == 0 and adj[t,k] == 0:
                # if rewired links have similar lengths
                if (np.abs(dists[s,t] - dists[k,t]) < eps and np.abs(dists[k,l] - dists[s,l]) < eps) or (
                    np.abs(dists[s,t] - dists[s,l]) < eps and np.abs(dists[k,l] - dists[k,t]) < eps):
                    adj[s,t] = 0
                    adj[t,s] = 0
                    adj[k,l] = 0
                    adj[l,k] = 0
                    adj[s,l] = 1
                    adj[l,s] = 1
                    adj[t,k] = 1
                    adj[k,t] = 1
                    edgeidcs[0][ed1] = s
                    edgeidcs[1][ed1] = l
                    edgeidcs[0][ed2] = t
                    edgeidcs[1][ed2] = k
                    it += 1
    return adj


import string
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