# Construct and Evaluate Climate Networks from Simulated Random Fields and ERA5 Data

This repository accompanies the paper “Pitfalls of Climate Network Construction - A Statistical Perspective” published at [Journal of Climate](https://journals.ametsoc.org/view/journals/clim/36/10/JCLI-D-22-0549.1.xml?tab_body=pdf). It is based on the [climnet](https://github.com/mlcs/climnet) repository.

## Installation

All experiments are implemented in Python 3.9.5.

Install all packages in `requirements.txt`.

Download this repository, move to the `climnet` directory and run `pip install -e .` to install your local version of the climnet package.


## Generating time series from an isotropic random field

If you want to generate time series from an isotropic Gaussian random field on a fixed grid, you can simply do the following:

```Python
import numpy as np
from climnet.myutils import *
from climnet.grid import FeketeGrid
from sklearn.gaussian_process.kernels import Matern

# generate the grid of your choice
grid = FeketeGrid(num_points = 1000)
# lon, lat are 1-dimensional arrays containing the coordinates of the grid points
lon, lat = grid.grid['lon'], grid.grid['lat']
# represent the grid in 3D space
cartesian_grid = spherical2cartesian(lon,lat)

# specify the lag-1 autocorrelation all grid points (here high lag-1 autocorrelation of 0.9)
ar_coeff = 0.9 * np.ones(1000)

# compute the covariance matrix on the grid based on the used kernel (here chordal Matern covariance)
kernel = 1.0 * Matern(length_scale=0.2, nu=1.5)
cov = kernel(cartesian_grid)

# sample time series from the isotropic Gaussian random field on the defined grid with the specified lag-1 autocorrelations
# data is an np.array of shape (n_time, num_gridpoints)
np.random.seed(18360)
data = diag_var_process(ar_coeff, cov, n_time=365)
```

## Introduction

The pipeline for any experiment is:

0)   Specify hyper parameters,
1)   Generate and save data,
2)   Calculate and save network statistics,
3)   Plot results.

Many scripts use a SLURM job array, where each job corresponds to an independent realization of the spatio-temporal random field.

After using a job array, run `compose_stats.py` to compose the stats computed in each run to a single file.

Before plotting, calculate network statistics of ground truth networks: `calc_true_pre.py`

## Reproducing the Experiments

First generate and IGRF data for reuse:
- Simulate an IGRF: `compute_data.py`
- Create a preprocessed data set from downloaded ERA5 data: `make_ds.py`

For parallelized, computationally expensive similarity calculations like the BIKSG Mutual Information estimator:
- Run `compute_empcorrs_parallel.py`
- Compose precalculated parts for the BIKSG mutual information estimator: `compose_BIKSG.py`

Table 1 (supplemental material):
- Local correlations for real data: `realbundle_corrinball.py`
- Bundle stats of real data: `realbundle_dens.py`
- Same for simulated data: `realbundle_sim.py`
- Random field statistics: `compute_randomfieldstats.py`

Fig 1:
- Visualise Matern IGRFs and sample paths: `igrf_visualisation.py`

Fig 2:
- Networks from lognormal data: `exp_nets.py`

Fig 3:
- FDR: `compute_fdr_parallel.py`
- Matrix distances: `compute_matrixdists_parallel.py`
- Plot networks and FDR for all similarity measures, and Fig. 3: `plot_allsims.py`

Fig 4:
- Calc and plot differences in resampled nets for real and sim. data: `densdiff.py`

Fig 5:
- Betweenness maps on a Gauss grid: `calc_betw_gaussgrid.py`

Fig 6:
- Real betweenness maps: `real_betw.py`

Fig 7:
- Compute graph statistics: `compute_graphstats_parallel.py`
- Plot graphstats: `plot_graphstats.py`

Fig 8 and 9:
- Stats of bundling behaviour: `compute_bundlestats_parallel.py` and `compute_otherbundlestats_parallel.py`
- Plot bundling stats: `plot_telestats.py`

Fig 10:
- Bundle stats of real data: `real_bundles.py`

Fig 11: 
- Compute and plot Spearman correlation and degrees given an anisotropic IGRF: `ar_exp.py`

Fig 12:
- Autocorrelation and degree maps of real networks: `real_nets.py`

Fig 13:
- Compute edge-wise IAAFT mean, std and quantiles: `arbias_bootstraps_parallel.py`
- Calculate correlation estimates given autocorrelated time series (Fig. 13a): `iaaft_arbias.py`
- Stats for bias corrections under anisotropic autocorr.: `signif_iaaft.py`
- IAAFT variance estimates and construct Fig. 13: `bootstraps_iaaft.py`

Fig 14:
- Networks from anisotropic noise: `noise_plots.py`



Fig 15:
- Network characteristics on anisotropic grids: `land_sea_bias.py`

Fig 16:
- Resampling statistics (not geomodel): `resampling_sim.py`
- Geomodel 2 adjacencies: `resampling_sim_geomodel.py`
- Compute graph statistics from Geomodel 2 adjacency: `georesampling_finish.py`
- Plot statistics of resampled networks: `plot_resampling.py`


## Additional information
- Compare Fekete to Fibonacci grid: `grid_comparison.py`


- Calculate Betweenness and Forman curvature histograms: `betweenness_sparse.py`


- Initialise all common variables (gets called in scripts): `grid_helper.py`


## Contact

If you have troubles running some of the experiments do not hesitate to get in touch.

## Citation

If you use this software please cite the following publication:

```bib
@article{PitfallsofClimateNetworkConstructionAStatisticalPerspective,
      author = "Moritz Haas and Bedartha Goswami and Ulrike von Luxburg",
      title = "Pitfalls of Climate Network Construction—A Statistical Perspective",
      journal = "Journal of Climate",
      year = "2023",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      volume = "36",
      number = "10",
      doi = "https://doi.org/10.1175/JCLI-D-22-0549.1",
      pages=      "3321 - 3342",
      url = "https://journals.ametsoc.org/view/journals/clim/36/10/JCLI-D-22-0549.1.xml"
}
```
