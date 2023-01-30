"""Climate network class."""
import sys
import os
import numpy as np
import xarray as xr
import scipy.stats as stat
import scipy.sparse as sparse
import scipy.special as special
from joblib import Parallel, delayed
import multiprocessing as mpi
from tqdm import tqdm

import climnet.link_bundles as lb
import climnet.event_synchronization as es
import climnet.grid as grid
from climnet.utils import holm
PATH = os.path.dirname(os.path.abspath(__file__))
RADIUS_EARTH = 6371  # radius of earth in km


def load(dataset, fname):
    """Load stored climate networks from .npz file.
    
    Parameters:
    -----------
    dataset: climnet.dataset.BaseDataset
        Dataset object.
    fname: str
        Filename of .npz file

    Returns:
    --------
    Net: climnet.net.BaseClimNet
        Climate network object.
    """
    with np.load(fname) as data:
        class_type = data['type'][0]

        # correlation based climate network
        if class_type == 'correlation':
            Net = CorrClimNet(
                dataset,
                corr_method=data['corr_method'][0],
                threshold=data['threshold'][0],
                confidence=data['confidence'][0],
                stat_test=data['stat_test'][0],
                posthoc_test=data['posthoc_test'][0]
            )
            Net.corr = data['corr']
            Net.pvalue = data['pvalue']

        # event synchronization network
        elif class_type == 'evs':
            Net = EventSyncClimNet(
                dataset,
                taumax=data['taumax'][0]
            )
        else:
            raise ValueError(f"Class type {class_type} not implemented!")
        Net.adjacency = data['adjacency']
        Net.lb = bool(data['lb'][0])

    return Net


class BaseClimNet:
    """ Climate Network class.
    Args:
    ----------
    dataset: BaseDataset object
        Dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.lb = False
        self.adjacency = None
        self.type = 'Base'

    def create(self):
        """Abstract create function."""
        return
    
    def save(self, fname):
        """Abstract save function."""
        return

    def check_densityApprox(self, adjacency, idx, nn_points_bw=1):
        """KDE approximation of the given adjacency and node idx."""
        coord_deg, coord_rad, map_idx = self.dataset.get_coordinates_flatten()
        all_link_idx_this_node = np.where(adjacency[idx, :] > 0)[0]
        link_coord = coord_rad[all_link_idx_this_node]
        bandwidth = self.obtain_kde_bandwidth(nn_points_bw=nn_points_bw)
        Z = lb.spherical_kde(link_coord, coord_rad, bandwidth)

        return {'z': Z, 'link_coord': link_coord, 'all_link': all_link_idx_this_node}

    def obtain_kde_bandwidth(self, nn_points_bw=1):
        """KDE bandwidth is set to nn_points_bw*max_dist_of_points.
        
        Parameters:
        ----------
        nn_points_bw: int
            Number of next-neighbor points of KDE bandwidth.
        """
        dist_eq = self.dataset.GridClass.get_distance_equator()
        bandwidth = nn_points_bw * dist_eq/RADIUS_EARTH

        return bandwidth

    def link_bundles(self, confidence, num_rand_permutations,
                     num_cpus=mpi.cpu_count(), nn_points_bw=1):
        """Significant test for adjacency. """
        # Get coordinates of all nodes
        coord_deg, coord_rad, map_idx = self.dataset.get_coordinates_flatten()

        # First compute Null Model of old adjacency matrix
        link_bundle_folder = PATH + f'/link_bundles/{self.dataset.var_name}/'
        null_model_filename = f'link_bundle_null_model_{self.dataset.var_name}'

        # Set KDE bandwidth to 2*max_dist_of_points
        dist_eq = self.dataset.GridClass.get_distance_equator()
        bandwidth = nn_points_bw * dist_eq/RADIUS_EARTH

        print("Start computing null model of link bundles!")
        lb.link_bundle_null_model(
            self.adjacency, coord_rad,
            link_bundle_folder=link_bundle_folder,
            filename=null_model_filename,
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus,
            bw=bandwidth
        )

        # Now compute again adjacency corrected by the null model of the link bundles
        try:
            print("Now compute new adjacency matrix!")
            adjacency = lb.link_bundle_adj_matrix(
                self.adjacency, coord_rad, link_bundle_folder, null_model_filename, 
                bw=bandwidth, perc=999, num_cpus=num_cpus
            )
        except:
            print("Other jobs for link bundling are not finished yet! Last job will do the rest!")
            sys.exit()
        self.adjacency = adjacency
        self.lb = True
        return adjacency

    def get_node_degree(self):
        node_degree = []
        for node in self.adjacency:
            node_degree.append(np.count_nonzero(node))

        return np.array(node_degree)

    def get_density(self, M = None):
        """Obtain density of adjacency matrix."""
        if M is None:
            M = self.adjacency
        density = (
            np.count_nonzero(M.flatten())
            / (M.shape[0]*(M.shape[0]-1))
        )
        print("Density of adjacency: ", density)
        return density

    def get_link_idx_lst(self, idx_lst):
        """Get list of links for a list of nodes.

        Parameters:
        ----------
        idx_lst: np.ndarray
            List of indices for the adjacency.

        Returns:
        --------
        List of link indices.
        """
        link_flat = np.zeros_like(self.adjacency[0, :], dtype=bool)
        for idx in idx_lst:
            link_flat = np.logical_or(
                link_flat, np.where(self.adjacency[idx, :] == 1, True, False)
            )
        return link_flat
  
    def ll_one_row(self, row, i, ):
        coord_i = self.dataset.get_map_index(i)
        idx_links = np.where(row == 1)[0]
        if len(idx_links) > 0:
            lon_links = []
            lat_links = []
            for j in idx_links:
                coord_j = self.dataset.get_map_index(j)
                lon_links.append(coord_j['lon'])
                lat_links.append(coord_j['lat'])

            ll_i = grid.haversine(coord_i['lon'], coord_i['lat'], 
                                  np.array(lon_links), np.array(lat_links),
                                  radius=RADIUS_EARTH)

        return ll_i

    def get_link_length_distribution_parallel(self):
        """Spatial link length distribution"""
        link_length = []

        backend = 'multiprocessing'
        num_cpus = mpi.cpu_count()
        link_length = (
                    Parallel(n_jobs=num_cpus, backend=backend)
                    (delayed(self.ll_one_row)
                     (row, i)
                     for i, row in enumerate(tqdm(self.adjacency))
                     )
        )

        return np.concatenate(link_length, axis=0)[:, -1]

    def get_link_length_distribution(self):
        """Spatial link length distribution"""
        link_length = []
        for i, row in enumerate(self.adjacency):
            coord_i = self.dataset.get_map_index(i)
            idx_links = np.where(row == 1)[0]

            if len(idx_links)>0:
                lon_links = []
                lat_links = []
                for j in idx_links:
                    coord_j = self.dataset.get_map_index(j)
                    lon_links.append(coord_j['lon'])
                    lat_links.append(coord_j['lat'])
                
                ll_i = grid.haversine(coord_i['lon'], coord_i['lat'],
                                      np.array(lon_links), np.array(lat_links),
                                      radius=RADIUS_EARTH)
                link_length.append(ll_i)
        
        return np.concatenate(link_length, axis=0)[:, -1]

    def convert2sparse(self, adjacency):
        """Convert adjacency matrix to scipy.sparce matrix.

        Args:
        -----
        adjacency: np.ndarray (N x N)
            Adjacency matrix of the network

        Returns:
        --------
        network: np.sparse
            The network a
        """
        network = sparse.csc_matrix(adjacency)
        print("Converted adjacency matrix to sparce matrix.")
        return network

    def save_single_matrix(self, matrix, fname):
        """Store network to file."""
        if os.path.exists(fname):
            print("Warning File" + fname + " already exists! No over writing!")
            os.rename(fname, fname+'_bak')
        sparse.save_npz(fname, matrix)
        print(f"Network stored to {fname}!")


class CorrClimNet(BaseClimNet):
    """Correlation based climate network.

    Parameters:
    -----------
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    threshold: float
        Default: -1.0
    stat_test: str
        Default: 'twosided'
    confidence: float
        Default: 0.99
    posthoc_test: str
        Method for statistical p-value testing, either 'bonf', 'dunn' or 'standard'. Default is 'bonf' (Holm-Bonferroni method)
    """

    def __init__(self, dataset, corr_method='spearman', 
                 threshold=-1.0, stat_test='twosided', confidence=0.99,
                 posthoc_test='bonf', k = None):
        super().__init__(dataset)
        # set to class variables
        self.type = 'correlation'
        self.corr_method = corr_method
        self.threshold = threshold
        self.confidence = confidence
        self.stat_test = stat_test
        self.k = k
        self.corr = None
        self.pvalue = None
        self.posthoc_test = posthoc_test

    def create(self, density = None):
        """Creates the climate network using a given correlation method."""
        # spearman's rank correlation
        if self.corr_method == 'spearman':
            self.corr, self.pvalue = self.calc_spearman(self.dataset, self.stat_test)
            if self.k is not None:
                # construct kNN graph
                # compute rowwise ranks
                abscorr0 = np.abs(self.corr)
                for i in range(len(abscorr0)):
                    abscorr0[i,i] = 0
                nn_idx = np.argsort(abscorr0, axis=1)[:,-self.k:]
                adj = np.zeros_like(self.corr)
                for i in range(len(adj)):
                    adj[i,nn_idx[i,:]] = 1 # still directed: j is k-nn to i
                self.adjacency = np.maximum(adj.T, adj)
                print(f'Constructed a {self.k}-NN graph.')
                return None
            elif density is not None:
                sorted_corr = np.sort(np.abs(self.corr).flatten())[:-self.corr.shape[0]]
                sorted_pval = np.sort(self.pvalue.flatten())[self.corr.shape[0]:]
                if self.posthoc_test == 'standard':
                    self.confidence = 1 - sorted_pval[int(density * len(sorted_pval) )] # corrected by possible links, TODO for bonf
                elif self.posthoc_test == 'bonf':
                    self.confidence = 1 - sorted_pval[int(density * len(sorted_pval))] * (len(sorted_pval) - int(density * len(sorted_pval)))
                self.threshold = sorted_corr[int(np.ceil((1-density) * len(sorted_corr)))]
                print('Set confidence to ', self.confidence, 'and threshold to ', self.threshold,'.')

            self.adjacency = self.get_adjacency(self.corr, self.pvalue,
                                                self.threshold, self.confidence,
                                                posthoc_test=self.posthoc_test)
        # pearson correlation
        elif self.corr_method == 'pearson':
            self.corr, self.pvalue = self.calc_pearson(self.dataset)
            if self.k is not None:
                # construct kNN graph
                # compute rowwise ranks
                abscorr0 = np.abs(self.corr)
                for i in range(len(abscorr0)):
                    abscorr0[i,i] = 0
                nn_idx = np.argsort(abscorr0, axis=1)[:,-self.k:]
                adj = np.zeros_like(self.corr)
                for i in range(len(adj)):
                    adj[i,nn_idx[i,:]] = 1 # still directed: j is k-nn to i
                self.adjacency = np.maximum(adj.T, adj)
                print(f'Constructed a {self.k}-NN graph.')
                return None
            elif density is not None:
                sorted_corr = np.sort(np.abs(self.corr).flatten())[:-self.corr.shape[0]]
                sorted_pval = np.sort(self.pvalue.flatten())[self.corr.shape[0]:]
                if self.posthoc_test == 'standard':
                    self.confidence = 1 - sorted_pval[int(density * len(sorted_pval))]
                elif self.posthoc_test == 'bonf':
                    self.confidence = 1 - sorted_pval[int(density * len(sorted_pval))] * (len(sorted_pval) - int(density * len(sorted_pval)))
                self.threshold = sorted_corr[int(np.ceil((1-density) * len(sorted_corr)))]
                print('Set confidence to ', self.confidence, 'and threshold to ', self.threshold,'.')
            self.adjacency = self.get_adjacency(self.corr, self.pvalue,
                                                self.threshold, 0, # 0 instead of self.confidence, because for fixed density, tends to be erroneous 1
                                                posthoc_test=self.posthoc_test)
        else:
            raise ValueError("Choosen correlation method does not exist!")

        return None

    def save(self, fname):
        """Store adjacency, correlation and pvalues to an .npz file.

        Parameters:
        -----------
        fname: str
            Filename to store .npz
        """
        if os.path.exists(fname):
            print("Warning File" + fname + " already exists! No over writing!")
            os.rename(fname, fname+'_bak')

        np.savez(fname,
                 corr=self.corr,
                 pvalue=self.pvalue,
                 adjacency=self.adjacency,
                 lb=np.array([self.lb]),
                 corr_method=np.array([self.corr_method]),
                 threshold=np.array([self.threshold]),
                 confidence=np.array([self.confidence]),
                 stat_test=np.array([self.stat_test]),
                 posthoc_test=np.array([self.posthoc_test]),
                 type=np.array([self.type])
                 )
        print(f"Network stored to {fname}!")
        return None

    def calc_spearman(self, dataset, test='onesided'):
        """Spearman correlation of the flattened and remove NaNs object.
        TODO: check dimension for spearman (Jakob)
        """
        data = dataset.flatten_array()
        print('Data shape: ', data.shape)

        corr, pvalue_twosided = stat.spearmanr(data, axis=0, nan_policy='propagate')

        if test == 'onesided':
            pvalue, self.zscore = self.onesided_test(corr)
        elif test == 'twosided':
            pvalue = pvalue_twosided
        else:
            raise ValueError('Choosen test statisics does not exist. Choose "onesided" '
                             + 'or "twosided" test.')
        
        print(f"Created spearman correlation matrix of shape {np.shape(corr)}")
        # print(np.all(corr.T == corr), np.all(pvalue.T == pvalue)) True, True
        return corr, pvalue

    def onesided_test(self, corr):
        """P-values of one sided t-test of spearman correlation.
        Following: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        """
        n = corr.shape[0]
        f = np.arctanh(corr)
        zscore = np.sqrt((n-3)/1.06) * f
        pvalue = 1 - stat.norm.cdf(zscore)

        return pvalue, zscore

    def calc_pearson(self, dataset):
        """Pearson correlation of the flattened array."""
        data = dataset.flatten_array()
        print(data.shape)
        # Pearson correlation
        corr = np.corrcoef(data.T)
        assert corr.shape[0] == data.shape[1]

        # get p-value matrix
        # TODO: Understand and check the implementation
        rf = corr[np.triu_indices(corr.shape[0], 1)]
        df = data.shape[1] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = special.betainc(0.5 * df, 0.5, df / (df + ts))
        p = np.zeros(shape=corr.shape)
        p[np.triu_indices(p.shape[0], 1)] = pf
        p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
        p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

        return corr, p

    def get_adjacency(self, corr, pvalue, threshold=0.5, confidence=0.95, posthoc_test='bonf'):
        """Create adjacency matrix from spearman correlation.

        Args:
        -----
        corr: np.ndarray (N x N)
            Spearman correlation matrix
        pvalue: np.ndarray (N x N)
            Pairwise pvalues of correlation matrix
        threshold: float
            Threshold to cut correlation
        confidence: float
            Confidence level
        posthoc_test: str
            type of statistical p-value testing
        Returns:
        --------
        adjacency: np.ndarray (N x N)
        """
        # Significance test on p-values
        if posthoc_test == "bonf" or posthoc_test == "dunn":
            pval_flat = pvalue.flatten()
            indices = holm(pval_flat, alpha=(1-confidence), corr_type=posthoc_test)
            mask_list = np.zeros_like(pval_flat)
            mask_list[indices] = 1
            mask_confidence = np.reshape(mask_list, pvalue.shape)
        elif posthoc_test == "standard":
            mask_confidence = np.where(pvalue <= (1-confidence), 1, 0)  # p-value test
        else:
            raise ValueError(f"Method {posthoc_test} does not exist!")

        # Threhold test on correlation values
        mask_threshold = np.where(np.abs(corr) >= threshold, 1, 0)
        # print('Symmetry of conf and thres:', np.all(mask_confidence.T == mask_confidence), np.all(mask_threshold.T == mask_threshold))
        # Because of Holm-Bonf one of two equal pvalues may be cut out
        adjacency = mask_confidence * mask_threshold
        # set diagonal to zero
        for i in range(len(adjacency)):
            adjacency[i, i] = 0
        adjacency = np.minimum(adjacency.T, adjacency)
        print("Created adjacency matrix.")

        return adjacency


class EventSyncClimNet(BaseClimNet):
    """Correlation based climate network.

    Parameters:
    -----------
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    threshold: float
        Default: -1.0
    stat_test: str
        Default: 'twosided'
    confidence: float
        Default: 0.99
    posthoc_test: str
        Method for statistical p-value testing. Default is 'bonf' (Holm-Bonferroni method)
    """

    def __init__(self, dataset, taumax=10):
        vars = dataset.vars
        if 'evs' not in vars:
            raise ValueError("For EventSyncNet a dataset has to be provided that contains event series!")
        super().__init__(dataset)
        # set to class variables
        self.type = 'evs'
        self.taumax = taumax
        self.es_filespath = PATH + '/es_files/'

    def save(self, fname):
        """Store adjacency, correlation and pvalues to an .npz file.

        Parameters:
        -----------
        fname: str
            Filename to store .npz
        """
        if os.path.exists(fname):
            print("Warning File" + fname + " already exists! Stored as backup")
            os.rename(fname, fname+'_bak')

        np.savez(fname,
                #  corr=self.corr,
                #  pvalue=self.pvalue,
                 adjacency=self.adjacency,
                 lb=np.array([self.lb]),
                 type=np.array([self.type]),
                 taumax=np.array([self.taumax])
                 )
        print(f"Network stored to {fname}!")
        return None

    def create(self, E_matrix_folder=None, null_model_file=None,
               num_jobs=1
               ):
        """
        This function has to be called twice, once, to compute the exact numbers of synchronous
        events between two time series, second again to compute the adjacency matrix 
        and the link bundles.
        Attention: The number of parrallel jobs that were used for the E_matrix needs to be 
        passed correctly to the function.
        """
        # Test if ES data is computed
        if self.dataset.ds['evs'] is None:
            raise ValueError("ERROR Event Synchronization data is not computed yet")
        else:
            data_evs = self.dataset.ds['evs']
            num_time_series = self.dataset.flatten_array().shape[1]
        if E_matrix_folder is None:
            E_matrix_folder = self.es_filespath + f'/E_matrix/{self.dataset.var_name}_{self.dataset.grid_step}/'
        else:
            self.es_filespath + f'/E_matrix/{E_matrix_folder}'

        if null_model_file is None:
            null_model_file = self.compute_es_null_model()
            sys.exit()
        else:
            null_model_file = self.es_filespath + 'null_model/' + null_model_file
            if not os.path.exists(null_model_file):
                raise ValueError(f'File {null_model_file} does not exist!')

        self.event_synchronization_run(data_evs=data_evs,
                                       E_matrix_folder=E_matrix_folder, taumax=self.taumax,
                                       null_model_file=null_model_file,
                                       )

        self.adjacency = self.compute_es_adjacency(E_matrix_folder=E_matrix_folder,
                                                   num_time_series=num_time_series)
        sparsity = np.count_nonzero(self.adjacency.flatten())/self.adjacency.shape[0]**2
        print(f"Sparsity of adjacency matrix: {sparsity}")

        return None

    def event_synchronization_run(self, data_evs, E_matrix_folder=None, taumax=10, min_sync_ev=1,
                                  null_model_file=None,
                                  sm='Jan', em='Dec'):
        """
        Event synchronization definition

        Parameters
        ----------
        data_evs: np.ndarray (lon, lat, timesteps)
            binary time series
        E_matrix_folder: str
            Folder where E-matrix files are to be stored
        taumax: int
            Maximum time delay between two events
        min_sync_ev: int
            Minimum number of sychronous events between two time series
        null_model_file: str
            File where null model for Adjacency is stored. If None, it will be computed again (costly!)
        Return
        ------
        None
        """

        # for job array
        try:
            min_job_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
            max_job_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
            job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
            run_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
            num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
            # num_jobs=max_job_id-min_job_id +1
            # num_jobs=30

            print((f"Run {run_id} for Subjob job_id: {job_id}/{num_jobs},"
                   f"Min Job ID: {min_job_id}, Max Job ID: {max_job_id}")
                  )
        except KeyError:
            job_id = 0
            num_jobs = 1
            run_id = 0
            print("Not running with SLURM job arrays, but with manual id: ", job_id)

        event_series_matrix = self.dataset.flatten_array(dataarray=data_evs, check=False).T

        if not os.path.exists(E_matrix_folder):
            os.makedirs(E_matrix_folder)
        E_matrix_filename = (f'E_matrix_{self.dataset.var_name}_'
                             f'q_{self.dataset.q}_min_num_events_{self.dataset.min_evs}_'
                             f'taumax_{taumax}_jobid_{job_id}.npy')

        if not os.path.exists(null_model_file):
            raise ValueError("Null model path does not exist {null_model_file}! ")

        if not os.path.exists(null_model_file):
            raise ValueError(f'Null model path does not exist! {null_model_file}')
        null_model = np.load(null_model_file)
        print('Null models shape: ', null_model.shape)

        print(f'JobID {job_id}: Start comparing all time series with taumax={taumax}!')
        if not os.path.exists(E_matrix_folder+E_matrix_filename):
            es.parallel_event_synchronization(event_series_matrix,
                                              taumax=taumax,
                                              min_num_sync_events=min_sync_ev,
                                              job_id=job_id,
                                              num_jobs=num_jobs,
                                              savepath=E_matrix_folder+E_matrix_filename,
                                              null_model=null_model,
                                              )
        else:
            print(f'File {E_matrix_folder+E_matrix_filename} does already exist!')

        path = E_matrix_folder
        E_matrix_files = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
        if len(E_matrix_files) < num_jobs:
            print(f"JobId {job_id}: Finished. Not all jobs have finished yet (missing {num_jobs} - {len(E_matrix_files)}!")
            sys.exit(0)

        return None

    def compute_es_adjacency(self, E_matrix_folder, num_time_series):
        if not os.path.exists(E_matrix_folder):
            raise ValueError("ERROR! The parallel ES is not computed yet!")

        adj_matrix_null_model = es.get_null_model_adj_matrix_from_E_files(E_matrix_folder,
                                                                          num_time_series,
                                                                          savepath=None)

        return adj_matrix_null_model
