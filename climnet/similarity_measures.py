# %%
from re import M
from numpy.core.fromnumeric import size
import scipy.spatial as ss
#import scipy.stats as sst
from scipy.signal import correlate
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log
import numpy as np
import warnings
from climnet.event_synchronization import event_synchronization
import scipy
#from statsmodels.tsa.arima_model import ARMA


# %%
#points x time
#dat = np.random.rand(1000,1000) > 0.95
#event_synchronization(dat)

# %%

# noise1, noise2 = np.random.randn(100,5), np.random.randn(100,5)
# ar1 = noise1
# ar2 = noise2
# for i in range(len(ar1)-1):
#     ar1[i+1] += 0.5 * ar1[i]
#     ar2[i+1] += 0.5 * ar2[i] + 0.1 * ar1[i+1]**2
# ## %%
# unb = HSIC(ar1,ar2)
# bia = HSIC(ar1,ar2,unbiased = False)
# bia[0]/unb[0], unb[0], bia[0], unb[2], bia[2]
#revised_mi(ar1,ar1, q=2)
# %%
def autocorr(X, t = 1):
    # uncertain how autocorr for multivariate data is originally computed, taking the mean is an arbitrary choice
    d = X.shape[1]
    return np.mean(np.corrcoef(X[:-t,:], X[t:,:],rowvar = False)[:d,d:])
# %%
#  D. Sejdinovic, A. Gretton and W. Bergsma.  A KERNEL TEST FOR THREE-VARIABLE INTERACTIONS, 2013.
#  median heuristic bandwidth selection

def median_heur(Z, num = 100):
    # Z of points x dimensions
    sizeZ = Z.shape[0]
    if len(Z.shape) == 1:
        Z = Z.reshape((-1,1))
    if sizeZ > num:
        Zmed = Z[np.random.permutation(sizeZ)[:num],:]
    else:
        Zmed = Z
    # compute sq distances betw. points
    G = (Zmed * Zmed).sum(axis = 1)
    Q = np.repeat(G.reshape((1,-1)), num, axis = 0)
    R = np.repeat(G.reshape((-1,1)), num, axis = 1)
    dists = R + Q -2* Zmed @ Zmed.T
    dists = dists - np.tril(dists)
    return np.sqrt(0.5 * np.median(dists[dists > 0]))

def rbf_dot(X,Y, sig):
    # compute sq distances betw. points
    lx = X.shape[0]
    ly = Y.shape[0]
    if len(X.shape) == 1:
        X = X.reshape((-1,1))
    if len(Y.shape) == 1:
        Y = Y.reshape((-1,1))
    G = (X*X).sum(axis = 1)
    H = (Y*Y).sum(axis = 1)
    Q = np.repeat(G.reshape((1,-1)), ly, axis = 0)
    R = np.repeat(H.reshape((-1,1)), lx, axis = 1)
    H = R + Q -2 * X @ Y.T
    return np.exp(-H / (2 * sig))

def estimate_head(X,Y):
    m = X.shape[0]

    autocorr = correlate(X+Y, X+Y)[:50,:50]
    for t in range(np.min(50,m)):
        ac = autocorr

def HSICstat(K,L):
    # implements unbiased estimator of Song 12 (instead of bias O(m**-1))
    m = len(K)
    Ktilde = K - np.diag(np.diag(K))
    Ltilde = L - np.diag(np.diag(L))
    onevec = np.ones(len(K))
    HSIC = (np.trace(Ktilde @ Ltilde) + onevec.T @ Ktilde @ onevec *
        onevec.T @ Ltilde @ onevec /( (m-1)*(m-2) ) - 2 * onevec.T @
        Ktilde @ Ltilde @ onevec / (m-2) ) / (m*(m-3))
    return HSIC


def HSIC(x,y, sigX = None, sigY = None, head = None, unbiased = True, permute = True):
    if len(x) != len(y):
        raise(Warning('x and y must have same length.'))

    m = len(x)
    if len(x.shape) == 1:
        x = x.reshape((-1,1))
    if len(y.shape) == 1:
        y = y.reshape((-1,1))
    
    if sigX is None:
        sigX = median_heur(x)
    if sigY is None:
        sigY = median_heur(y)
    
    tail = m
    head = m
    #take first autocorr < 0.2 to be almost indep after shift
    xy = x + y
    for i in range(50):
        if autocorr(xy, i+1) < 0.2:
            head = i+1
            break

    if head > min(75,m):
        warnings.warn('Possibly long memory process, the output of test might be FALSE.')

    head = min(50,head)
    K = rbf_dot(x,x,sigX)
    L = rbf_dot(y,y,sigY)
    #HSIC = HSICstat(K,L)
    if unbiased:
        HSIC = HSICstat(K,L)
    else:
        H = np.eye(m) - 1/ m * np.ones((m,m))
        Kc = H @ K @ H # symmetric anyways
        HSIC = m * np.mean(Kc * L) # computes the trace of HKHL /m**2


    if permute:
        # bootstrap by shifting one of the time series
        nullApprox = np.zeros(tail-head)    
        for i in range(tail-head):
            idcs = np.concatenate((np.arange(i + head,tail),np.arange(0,i+head)))
            if unbiased:
                nullApprox[i] = HSICstat(K,L[np.ix_(idcs,idcs)])
            else:
                nullApprox[i] = m * np.mean(Kc * L[np.ix_(idcs,idcs)])
        return HSIC, nullApprox, np.mean(nullApprox > HSIC)
    return HSIC

# %%
def vd(d,q):
    # Compute the volume of unit l_q ball in d dimensional space
    if (q==float('inf')):
        return d*log(2)
    elif (q == 2) and (d==1):
        return log(2)
    return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))


'''
Use revised_mi or Mixed_KSG for MI estimation.

hoeffding is a traditional independence test for continuous distributions.

ShiftHSIC is the only one for non iid data! (Kernel based)

Using the Wasserstein distance is just an idea as is seems to have more desirable properties than KL, but HSIC does smth similar.
'''


# MI estimator for discrete-continuous mixtures
# https://github.com/wgao9/mixed_KSG

def Mixed_KSG(x,y,k=5):
    '''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using *Mixed-KSG* mutual information estimator
		Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
		y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
		k: k-nearest neighbor parameter
		Output: one number of I(X;Y)
	'''
    assert (len(x)==len(y)), "Lists should have same length"
    assert (k <= len(x)-1), "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N,1))
    dx = len(x[0])   	
    if y.ndim == 1:
        y = y.reshape((N,1))
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
    return ans


# (improved) KSG estimator for MI
# https://github.com/wgao9/knnie

# old estimator
def kraskov_mi(x,y,k=5):
    '''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using KSG mutual information estimator
		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		Output: one number of I(X;Y)
	'''
    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])   	
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans_xy = -digamma(k) + digamma(N) + (dx+dy)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
    ans_x = digamma(N) + dx*log(2)
    ans_y = digamma(N) + dy*log(2)
    for i in range(N):
        ans_xy += (dx+dy)*log(knn_dis[i])/N
        ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
        ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N
        
    return ans_x+ans_y-ans_xy


def revised_mi(x,y,k=5,q=float('inf')):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)
        Input: x: 2D list of size N*d_x
        y: 2D list of size N*d_y
        k: k-nearest neighbor parameter
        q: l_q norm used to decide k-nearest distance
        Output: one number of I(X;Y)
    '''

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if len(x.shape) == 1:
        x= x.reshape((x.shape[0],1))
    if len(y.shape) == 1:
        y= y.reshape((y.shape[0],1))

    dx = len(x[0,:])   	
    dy = len(y[0,:])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=q)[0][k] for point in data]
    ans_xy = -digamma(k) + log(N) + vd(dx+dy,q)
    ans_x = log(N) + vd(dx,q)
    ans_y = log(N) + vd(dy,q)
    for i in range(N):
        ans_xy += (dx+dy)*log(np.maximum(knn_dis[i],1e-15))/N
        ans_x += -log(len(tree_x.query_ball_point(x[i],np.maximum(knn_dis[i],1e-15),p=q))-1)/N+dx*log(np.maximum(knn_dis[i],1e-15))/N
        ans_y += -log(len(tree_y.query_ball_point(y[i],np.maximum(knn_dis[i],1e-15),p=q))-1)/N+dy*log(np.maximum(knn_dis[i],1e-15))/N		
    return ans_x+ans_y-ans_xy


# traditional for continuous variables:
#from  XtendedCorrel import hoeffding

#hoeffding(x,y)

# or using pandas dataframe df
#df.corr(method=hoeffding)

# propose Wasserstein information estimator:
# estimate W(P ^ (X,Y), P^X x P^Y) = sup_(f in Lip1) E f(P ^ (X,Y)) - E f(P^X x P^Y)

'''
# so train WGAN critic for every pair of points -> too computationally expensive -> only train few epochs because, close points still similar enough
# or try http://www.kernel-operations.io/geomloss/index.html
import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
#  N.B.: We use float64 numbers to get nice limits when blur -> +infinity
dtype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
loss = SamplesLoss("sinkhorn", p=1, blur=0.01) # play around with blur
L = loss(x, y)

# or
from layers import SinkhornDistance
sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
dist, P, C = sinkhorn(x, y)


# maybe better
# but kernel based (MMD) is also an IPM with good conv. rates so should perform similarly.
import ot
a, b = np.ones((n,)) / n, np.ones((n,)) / n
ot.sliced_wasserstein_distance(x, y, a, b, n_projections, seed)
ot.dist(x,y)
'''
# %%
# pyunicorn.climate.mutual_info.calculate_similarity_measure(data)


def calculate_mutual_information(anomaly, n_bins=None,silence_level = 2):
        """
        Calculate the mutual information matrix at zero lag.
        .. note::
           Slow since solely based on Python and Numpy!
        :type anomaly: 2D array (time, index)
        :arg anomaly: The anomaly time series.
        :arg int n_bins: The number of bins for estimating probability
                     distributions.
        :rtype: 2D array (index, index)
        :return: the mutual information matrix at zero lag.
        """
        if silence_level <= 1:
            print("Calculating mutual information matrix at zero lag from "
                  "anomaly values...")

        #  Define references to numpy functions for faster function calls
        histogram = np.histogram
        histogram2d = np.histogram2d
        log = np.log

        #  Normalize anomaly time series to zero mean and unit variance
        #self.data.normalize_time_series_array(anomaly)

        #  Get faster reference to length of time series = number of samples
        #  per grid point.
        n_samples, N = anomaly.shape

        if n_bins is None:
            n_bins = int(np.floor(np.sqrt(n_samples/5)))
        #  Initialize mutual information array
        mi = np.zeros(( N,  N))

        #  Get common range for all histograms
        range_min = anomaly.min()
        range_max = anomaly.max()

        #  Calculate the histograms for each time series
        p = np.zeros(( N, n_bins))

        for i in range( N):
            p[i, :] = histogram(
                anomaly[:, i], bins=n_bins, range=(range_min, range_max)
            )[0].astype("float64")

        #  Normalize by total number of samples = length of each time series
        p /= n_samples

        #  Make sure that bins with zero estimated probability are not counted
        #  in the entropy measures.
        p[p == 0] = 1

        #  Compute the information entropies of each time series
        H = - (p * log(p)).sum(axis=1)

        # Initialize progress bar
        if  silence_level <= 1:
            progress = progressbar.ProgressBar(maxval= N**2).start()

        #  Calculate only the lower half of the MI matrix, since MI is
        #  symmetric with respect to X and Y.
        for i in range( N):
            # Update progress bar every 10 steps
            if  silence_level <= 1:
                if (i % 10) == 0:
                    progress.update(i**2)

            for j in range(i):
                #  Calculate the joint probability distribution
                pxy = histogram2d(
                    anomaly[:, i], anomaly[:, j], bins=n_bins,
                    range=((range_min, range_max),
                           (range_min, range_max)))[0].astype("float64")

                #  Normalize joint distribution
                pxy /= n_samples

                #  Compute the joint information entropy
                pxy[pxy == 0] = 1
                HXY = - (pxy * log(pxy)).sum()

                #  ... and store the result
                mi.itemset((i, j), H.item(i) + H.item(j) - HXY)
                mi.itemset((j, i), mi.item((i, j)))

        if  silence_level <= 1:
            progress.finish()

        return mi






def lw_analytical_shrinkage_estimator(X,k=None):
    # translated from: https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html#Programming_Code
    # X is the raw data matrix of size n x p:
    # - the rows correspond to observations
    # - the columns correspond to variables

    # If the second (optional) parameter k is absent, not-a-number, or empty, 
    # the algorithm demeans the data by default, and then adjusts 
    # the effective sample size accordingly by subtracting one.

    # If the user inputs k = 0, then no demeaning takes place and
    # the effective sample size remains n.

    # If the user inputs k >= 1, then it signifies that the data X 
    # has already been demeaned or otherwise pre-processed; for example,
    # the data might constitute OLS residuals based on a linear regression
    # model with k regressors. No further demeaning takes place then,
    # but the effective sample size is adjusted accordingly by subtracting k.
    n,p = X.shape
    if k is None:
        X -= X.mean(axis = 0)
        k = 1

    n = n-k
    eigvals, eigvecs = scipy.linalg.eigh((X.T @ X + 1e-10 * np.eye(p)) / n) # for stability
    eigvals, eigvecs = eigvals[np.argsort(eigvals)], eigvecs[:,np.argsort(eigvals)]

    # compute analytical nonlinear shrinkage kernel formula
    eigvals = eigvals[np.maximum(0,p-n):p]
    L = np.tile(eigvals.reshape((-1,1)), (1, np.minimum(n,p)) )
    h = n ** (-1/3)
    H = h * L.T
    x = (L-L.T) / H
    ftilde = 3/(4*np.sqrt(5)) * (np.maximum(np.zeros_like(x), 1-x ** 2 / 5) / H).mean(axis = 1)
    Hftemp = -3/(10 * np.pi) * x + 3 /( 4 * np.sqrt(5) * np.pi) * (1-x ** 2 / 5) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    Hftemp[np.abs(x) == np.sqrt(5)] = -3/(10 * np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = (Hftemp / H).mean(axis = 1)
    if p <= n:
        dtilde = eigvals / ((np.pi * p/n * eigvals * ftilde) ** 2 + (1 - p/n - np.pi * p/n * eigvals * Hftilde) ** 2)
    else:
        Hftilde0 = 1/np.pi * ( 3/(10 * h ** 2) + 3 / (4 * np.sqrt(5) * h) *(1-1/(5 * h ** 2)) * np.log((1+np.sqrt(5)*h)/(1-np.sqrt(5)*h)) ) * np.mean(1/eigvals)
        dtilde0 = 1 / (np.pi * (p-n)/n * Hftilde0)
        dtilde1 = eigvals / (np.pi ** 2 * eigvals ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate((dtilde0 * np.ones(p-n), dtilde1))
    if dtilde.dtype != np.float64:
        dtilde = dtilde.view(np.float64)[::2] #reduces complex dtilde with 0j...
    sigmatilde = eigvecs @ np.diag(dtilde) @ eigvecs.T
    sigmatilde = (sigmatilde.T + sigmatilde) / 2 # ensure symmetry for stability
    return sigmatilde, dtilde # eigvals in dtilde not sorted! error?