"""Create land sea mask."""

# %%
import os, sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
PATH = os.path.dirname(os.path.abspath(__file__))

# %%
dirname = "/home/jakob/mount_volume/era5/pSurf_Tair/1997"
fname = dirname +"/PSurf_WFDE5_CRU_199701_v1.0.nc" 

ds = xr.open_dataset(fname)
da = ds['PSurf']

# %%
mask = np.ones_like(da[0].data)
mask[np.isnan(da[0].data)] = np.NaN
mask_da = xr.DataArray(mask, coords=[da.lat, da.lon])
mask_ds = mask_da.to_dataset(name='lsm') 

# %%
# Store
outfile = PATH + '/../input/land-sea-mask_era5.nc'
mask_ds.to_netcdf(outfile)

# %%
