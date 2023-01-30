""" Base class for the different dataset classes of the multilayer climate network."""

from math import exp
import sys,os
import numpy as np
import scipy.interpolate as interp
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as ctp

import climnet.grid as grid

    
class BaseDataset:
    """ Base Dataset.
    Args:
    
    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        List of time range, e.g. ['1997-01-01', '2019-01-01']. Default: None
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'regular',
        'fekete', 'fibonacci', 'fib_old', 'fib_maxmin', 'gaussian'
    month_range: list
        List of month range, e.g. ["Jan", "Dec"]. Default: None
    lsm: bool
        Default:False
    **kwargs
    """

    def __init__(self,
                 var_name,
                 data_nc=None,
                 load_nc=None,
                 time_range=None,
                 month_range=None,
                 lon_range=[-180, 180],
                 lat_range=[-90, 90],
                 grid_step=1,
                 grid_type='regular',
                 num_iter = 1000,
                 epsilon = 0.36,
                 lsm=False,
                 cut = False,
                 climatology = None,
                 **kwargs
                 ):

        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")

        # initialized dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                print(f"You are here: {PATH}!")
                raise ValueError(f"File does not exist {data_nc}!")

            ds = xr.open_dataset(data_nc)

            ds = self.check_dimensions(ds)  # check dimensions

            self.time_range = time_range
            self.grid_step = grid_step
            self.grid_type = grid_type
            self.lsm = lsm
            self.info_dict = kwargs

            # choose time range
            if self.time_range is not None:
                ds = self.interp_times(ds, self.time_range)
            if month_range is not None:
                ds = self.get_month_range_data(ds, start_month=month_range[0],
                                               end_month=month_range[1])

            # regridding
            self.GridClass = self.create_grid(grid_type=self.grid_type, num_iter=num_iter, epsilon = epsilon)
            if cut:
                self.grid = self.GridClass.cut_grid(
                    [ds['lat'].min().data, ds['lat'].max().data], 
                    [ds['lon'].min().data, ds['lon'].max().data] 
                )
            else:
                self.grid = self.GridClass.grid

            # da_list = []
            # for vname, da in ds.data_vars.items():
            #     print("Variables in dataset: ", vname)
            da = ds[var_name]
            da = self.interp_grid(da, self.grid)
            
            if self.lsm is True:
                self.mask, da = self.get_land_sea_mask_data(da)
            else:
                self.mask = xr.DataArray(
                                data=np.ones_like(da[0].data),
                                dims=da.sel(time=da.time[0]).dims,
                                coords=da.sel(time=da.time[0]).coords,
                                name='mask')

            # da_list.append(da)

            # self.ds = xr.merge(da_list) # merge regridded dataarrays
            self.ds = da.to_dataset(name=var_name)
            if climatology == 'dayofyear':
                anomalies = (da.groupby(f"time.dayofyear")
                     - da.groupby(f"time.dayofyear").mean("time"))
                print(f'Created dayofyearly anomalies!')
                self.ds['anomalies'] = anomalies

        # load dataset object from file
        elif load_nc is not None:
            self.grid_step = grid_step
            self.grid_type = grid_type
            self.lsm = lsm
            self.load(load_nc)

        # select a main var name
        self.vars = []
        for name, da in self.ds.data_vars.items():
            self.vars.append(name)
        self.var_name = var_name if var_name is not None else self.vars[0]

        # Flatten index in map
        self.indices_flat, self.idx_map = self.init_mask_idx()

        if load_nc is None:
            self.ds = self.ds.assign_coords(idx_flat=("points", self.idx_map))

    def load(self, load_nc):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset
        
        """
        # check if file exists
        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            print(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        ds = xr.open_dataset(load_nc)

        self.time_range = [ds.time.data[0], ds.time.data[-1]]
        self.lon_range = [float(ds.lon.min()), float(ds.lon.max())]
        self.lat_range = [float(ds.lat.min()), float(ds.lat.max())]

        #self.grid_step = ds.attrs['grid_step']
        #self.grid_type = ds.attrs['grid_type']
        #self.lsm = bool(ds.attrs['lsm'])
        #self.info_dict = ds.attrs  # TODO
        # Read and create grid class
        self.grid = dict(lat=ds.lat.data, lon=ds.lon.data)
        if self.grid_type == 'gaussian':
            self.GridClass = grid.GaussianGrid(self.grid_step, self.grid_step,
                                               grid=self.grid)
        elif self.grid_type == 'fibonacci':
            dist_equator = grid.degree2distance_equator(self.grid_step)
            self.GridClass = grid.FibonacciGrid(dist_equator, grid=self.grid)
        elif self.grid_type == 'fekete':
            dist_equator = grid.degree2distance_equator(self.grid_step)
            self.num_points = grid.get_num_points(dist_equator)
            try:
                self.num_iter = ds.attrs['num_iter']
                self.GridClass = grid.Fekete(self.num_points, self.num_iter, grid=self.grid)
            except:
                self.GridClass = grid.Fekete(self.num_points, grid=self.grid)
        else:
            raise ValueError('Grid type does not exist.')

        for name, da in ds.data_vars.items():
            print("Variables in dataset: ", name)

        # points which are always NaN will be NaNs in mask
        mask = np.ones_like(ds[name][0].data, dtype=bool)
        for idx, t in enumerate(ds.time):
            mask *= np.isnan(ds[name].sel(time=t).data)

        self.mask = xr.DataArray(
                        data=xr.where(mask == False, 1, np.NaN),
                        dims=da.sel(time=da.time[0]).dims,
                        coords=da.sel(time=da.time[0]).coords,
                        name='lsm')

        self.ds = ds

        return None

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            print("File" + filepath + " already exists!")
            os.rename(filepath, filepath + "_backup")

        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            **self.info_dict
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)
        return None

    def check_dimensions(self, ds):
        """
        Checks whether the dimensions are the correct ones for xarray!
        """
        lon_lat_names = ['longitude', 'latitude']
        xr_lon_lat_names = ['lon', 'lat']
        dims = list(ds.dims)

        for idx, lon_lat in enumerate(lon_lat_names):
            if lon_lat in dims:
                print(dims)
                print(f'Rename:{lon_lat} : {xr_lon_lat_names[idx]} ')
                ds = ds.rename({lon_lat: xr_lon_lat_names[idx]})
                dims = list(ds.dims)
                print(dims)
        clim_dims = ['time', 'lat', 'lon']
        for dim in clim_dims:
            if dim not in dims:
                raise ValueError(f"The dimension {dim} not consistent with required dims {clim_dims}!")

        # If lon from 0 to 360 shift to -180 to 180
        if max(ds.lon) > 180:
            print("Shift longitude!")
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

        return ds

    def create_grid(self, grid_type='regular', num_iter = 1000, epsilon = 0.36):
        """Common grid for all datasets.

        ReturnL
        -------
        Grid: grid.BaseGrid
        """
        dist_equator = grid.degree2distance_equator(self.grid_step)
        if grid_type == 'gaussian':
            Grid = grid.GaussianGrid(self.grid_step, self.grid_step)
        elif grid_type == 'fekete':
            num_points = grid.get_num_points(dist_equator)
            Grid = grid.FeketeGrid(num_points, num_iter)
        elif grid_type == 'fibonacci':
            Grid = grid.FibonacciGrid(dist_equator, epsilon = epsilon)
        elif grid_type == 'fib_old':
            Grid = grid.FibonacciGrid(dist_equator, 'old')
        elif grid_type == 'fib_maxmin':
            Grid = grid.FibonacciGrid(dist_equator, 'maxmin')
        elif grid_type == 'regular':
            Grid = grid.RegularGrid(self.grid_step, self.grid_step)
        elif isinstance(grid_type, list): # grid_type = [grid_step_lon, grid_step_lat, grid]
            Grid = grid.RegularGrid(grid_type[0], grid_type[1], grid_type[2])
        else:
            raise ValueError('Grid type does not exist.')

        return Grid

    def interp_grid(self, dataarray, new_grid):
        """Interpolate dataarray on new grid.
        dataarray: xr.DataArray
            Dataarray to interpolate.
        new_grid: dict
            Grid we want to interpolate on.
        """
        new_points = np.array([new_grid['lon'], new_grid['lat']]).T

        lon_mesh, lat_mesh = np.meshgrid(dataarray.lon, dataarray.lat)
        origin_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        # for one timestep
        if len(dataarray.data.shape) < 3:
            origin_values = dataarray.data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(origin_points, origin_values, new_points,
                                         method='nearest')
            new_values = np.array(new_values).T
            coordinates = dict(points=np.arange(0, len(new_points), 1),
                               lon=("points", new_points[:, 0]),
                               lat=("points", new_points[:, 1]))
            dims = ['points']
        else:
            new_values = []
            for idx, t in enumerate(dataarray.time):
                origin_values = dataarray.sel(time=t.data).data.flatten()
                new_values.append(
                    interp.griddata(origin_points, origin_values,
                                    new_points,
                                    method='nearest')
                )
            coordinates = dict(time=dataarray.time.data,
                               points=np.arange(0, len(new_points), 1),
                               lon=("points", new_points[:, 0]),
                               lat=("points", new_points[:, 1]))
            dims = ['time', 'points']
            new_values = np.array(new_values)

        new_dataarray = xr.DataArray(
                            data=new_values,
                            dims=dims,
                            coords=coordinates,
                            name=dataarray.name)

        return new_dataarray

    def cut_map(self, lon_range, lat_range):
        """Cut an area in the map.

        Args:
        ----------
        lon_range: list [min, max]
            range of longitudes
        lat_range: list [min, max]
            range of latitudes

        Return:
        -------
        ds_area: xr.dataset
            Dataset cut to range
        """
        ds_cut = self.ds.sel(
            lon=slice(np.min(lon_range), np.max(lon_range)),
            lat=slice(np.min(lat_range), np.max(lat_range))
        )
        return ds_cut

    def get_land_sea_mask_data(self, dataarray):
        """
        Compute a land-sea-mask for the dataarray,
        based on an input file for the land-sea-mask.
        """
        PATH = os.path.dirname(os.path.abspath(__file__))  # Adds higher directory 
        lsm_mask_ds = xr.open_dataset(PATH + "/../input/land-sea-mask_era5.nc")
        lsm_mask = self.interp_grid(lsm_mask_ds['lsm'], self.grid)

        land_dataarray = xr.where(np.array([lsm_mask]) == 1, dataarray, np.nan)
        return lsm_mask, land_dataarray

    def flatten_array(self, dataarray=None, time=True, check=True):
        """Flatten and remove NaNs.
        """
        if dataarray is None:
            dataarray = self.ds[self.var_name]

        idx_land = np.where(self.mask.data.flatten() == 1)[0]
        if time is False:
            buff = dataarray.data.flatten()
            buff[np.isnan(buff)] = 0.0  # set missing data to climatology
            data = buff[idx_land]
        else:
            data = []
            for idx, t in enumerate(dataarray.time):
                buff = dataarray.sel(time=t.data).data.flatten()
                buff[np.isnan(buff)] = 0.0  # set missing data to climatology
                data.append(buff[idx_land])

        # check
        if check is True:
            num_nonzeros = np.count_nonzero(data[-1])
            num_landpoints = sum(~np.isnan(self.mask.data.flatten()))
            print(f"The number of non-zero datapoints {num_nonzeros} "
                  + f"should approx. be {num_landpoints}.")

        return np.array(data)

    def init_mask_idx(self):
        """
        Initializes the flat indices of the map.
        Usefule if get_map_index is called multiple times.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')
        return indices_flat, idx_map

    def get_map(self, data, name=None):
        """Restore dataarray map from flattened array.

        TODO: So far only a map at one time works, extend to more than one time

        This also includes adding NaNs which have been removed.
        Args:
        -----
        data: np.ndarray (n,0)
            flatten datapoints without NaNs
        mask_nan: xr.dataarray
            Mask of original dataarray containing True for position of NaNs
        name: str
            naming of xr.DataArray

        Return:
        -------
        dmap: xr.dataArray
            Map of data
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(mask_arr) == len(data)

        # create array with NaNs
        data_map = np.empty(len(mask_arr)) 
        data_map[:] = np.NaN

        # fill array with sample
        data_map[mask_arr] = data

        dmap = xr.DataArray(
                data=data_map,
                dims=['points'],
                coords=dict(points=self.ds.points.data, 
                            lon=("points", self.ds.lon.data),
                            lat=("points", self.ds.lat.data)),
                name=name)

        return dmap

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of flatten array 
           without Nans.

        # Attention: Mask has to be initialised

        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs

        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        indices_flat = self.indices_flat

        idx_map = self.idx_map

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        if idx_flat > len(indices_flat):
            raise ValueError("Index doesn't exist.")
        map_idx = {
            'lat': buff.lat.data,
            'lon': buff.lon.data,
            'idx': np.argwhere(idx_map.data == idx_flat)
        }
        return map_idx

    def get_coordinates_flatten(self):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad: 
        map_idx:
        """
        # length of the flatten array with NaNs removed
        length = self.flatten_array().shape[1]

        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff['lat'][0], buff['lon'][0]])
            map_idx.append(buff['idx'][0])

        coord_rad = np.radians(coord_deg)

        return np.array(coord_deg), coord_rad, map_idx

    def get_index_for_coord(self, lat, lon):
        """Get index of flatten array for specific lat, lon."""
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')

        idx = idx_map.sel(lat=lat, lon=lon, method='nearest')
        if np.isnan(idx):
            print("Warning the lon lat is not defined!")
        return int(idx)

    def flat_idx_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        full_idx_lst = np.zeros(len_index)
        full_idx_lst[idx_list] = 1

        return full_idx_lst

    def find_nearest(self, a, a0):
        """
        Element in nd array `a` closest to the scalar value `a0`
        ----
        Args a: nd array
             a0: scalar value
        Return
            idx, value
        """
        idx = np.abs(a - a0).argmin()
        return idx, a.flat[idx]

    def interp_times(self, dataset, time_range):
        """Interpolate time in time range in steps of days.
        TODO: So far only days works.
        """
        time_grid = np.arange(
            time_range[0], time_range[1], dtype='datetime64[D]'
        )
        ds = dataset.interp(time=time_grid, method='nearest')
        return ds

    def get_data_timerange(self, data, time_range):
        start_time, end_time = time_range
        print(f"Get data within timerange: {start_time} - {end_time}")

        var_data_timerange = data.sel(time=slice(start_time, end_time))
        print("Done!")

        return var_data_timerange

    def _get_index_of_month(self, month):
        idx = -1
        idx = self.months.index(month)
        if idx == -1:
            print("This month does not exist: ", month)
            sys.exit(1)
        return idx

    def _is_in_month_range(self, month, start_month, end_month):
        start_month_idx = self._get_index_of_month(start_month)+1
        end_month_idx = self._get_index_of_month(end_month)+1

        if start_month_idx <= end_month_idx:
            mask = (month >= start_month_idx) & (month <= end_month_idx)
        else:
            mask = (month >= start_month_idx) | (month <= end_month_idx)
        return mask

    def get_month_range_data(self, dataset, start_month='Jan', end_month='Dec'):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        seasonal_data = dataset.sel(time=self._is_in_month_range(dataset['time.month'], start_month, end_month))
        return seasonal_data

    def get_mean_loc(self, idx_lst):
        """
        Gets a mean location for a list of indices
        """
        lon_arr = []
        lat_arr = []
        for idx in idx_lst:
            map_idx = self.get_map_index(idx)
            lon_arr.append(map_idx['lon'])
            lat_arr.append(map_idx['lat'])
        mean_lat = np.mean(lat_arr)

        if max(lon_arr)-min(lon_arr) > 180:
            lon_arr = np.array(lon_arr)
            lon_arr[lon_arr < 0] = lon_arr[lon_arr < 0]+360

        mean_lon = np.mean(lon_arr)
        if mean_lon > 180:
            mean_lon -= 360
        return(mean_lat, mean_lon)

    def get_locations_in_range(self, lon_range, lat_range, def_map):
        """
        Returns a map with the location within certain range.

        Parameters:
        -----------
        lon_range: list
            Range of longitudes, i.e. [min_lon, max_lon]
        lat_range: list
            Range of latitudes, i.e. [min_lat, max_lat]
        def_map: xr.Dataarray
            Map of data, i.e. the mask.

        Returns:
        --------
        idx_lst: np.array
            List of indices of the flattened map.
        mmap: xr.Dataarray
            Dataarray including ones at the location and NaNs everywhere else
        """
        if (max(lon_range) - min(lon_range) < 180):
            mask = (
                (def_map['lat'] >= min(lat_range))
                & (def_map['lat'] <= max(lat_range))
                & (def_map['lon'] >= min(lon_range))
                & (def_map['lon'] <= max(lon_range))
                )
        else:   # To account for areas that lay at the border of -180 to 180
            mask = (
                (def_map['lat'] >= min(lat_range))
                & (def_map['lat'] <= max(lat_range))
                & ( (def_map['lon'] <= min(lon_range)) | (def_map['lon'] >= max(lon_range)))
                )
        mmap = xr.where(mask, def_map, np.nan)
        idx_lst = np.where(self.flatten_array(mmap, time=False, check=False) == 1)[0]

        return idx_lst, mmap

    def use_time_snippets(self, time_snippets):
        """Cut time snippets from dataset and concatenate them.

        Parameters:
        -----------
        time_snippets: np.datetime64  (n,2)
            Array of n time snippets with dimension (n,2).

        Returns:
        --------
        xr.Dataset with concatenate times
        """
        ds_lst = []
        for time_range in time_snippets:
            ds_lst.append(self.ds.sel(time=slice(time_range[0], time_range[1])))

        self.ds = xr.merge(ds_lst)

        return self.ds

    ##############################################################################
    # Plotting routines from here on
    ##############################################################################
    def plot_map(self, dmap, plot_type='scatter', central_longitude=0, central_latitude = 0,
                 vmin=None, vmax=None, color='RdBu', bar=True,
                 fig=None, ax=None, 
                 ctp_projection="Mollweide",
                 label=None, title=None, significant_mask=False):
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
        
            # lat and lon labels
            gl.ylabels_right = False
            gl.top_labels = False
            gl.bottom_labels = False
        
        projection = ccrs.PlateCarree(central_longitude=central_longitude)

        # set colormap
        cmap = plt.get_cmap(color)
        kwargs_pl = dict()  # kwargs plot function
        kwargs_cb = dict()  # kwargs colorbar 
        if bar == 'discrete':
            normticks = np.arange(0, dmap.max(skipna=True)+2, 1)
            kwargs_pl['norm'] = mpl.colors.BoundaryNorm(normticks, cmap.N)
            kwargs_cb['ticks'] = normticks + 0.5

        # areas which are dotted
        if significant_mask:
            left_out = xr.where(np.isnan(self.mask), 1, np.nan)
            ax.contourf(dmap.coords['lon'], dmap.coords['lat'],
                        left_out, 2, hatches=['...', '...'], colors='none', extend='lower',
                        transform=projection)
        
        # plotting
        if plot_type =='scatter':
            im = ax.scatter(x=dmap.coords['lon'], y=dmap.coords['lat'],
                            c=dmap.data, vmin=vmin, vmax=vmax, cmap=cmap,
                            transform=projection)

        elif plot_type == 'colormesh':
            # interpolate grid of points to regular grid
            lon_interp = np.arange(dmap.coords['lon'].min(),
                                   dmap.coords['lon'].max() + self.grid_step,
                                   self.grid_step)
            lat_interp = np.arange(dmap.coords['lat'].min(),
                                   dmap.coords['lat'].max() + self.grid_step,
                                   self.grid_step)

            lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
            new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
            origin_points = np.array([dmap.coords['lon'], dmap.coords['lat']]).T
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            new_values = interp.griddata(origin_points, dmap.data, new_points,
                                         method='nearest')
            mesh_values = new_values.reshape(len(lat_interp), len(lon_interp))

            im = ax.pcolormesh(
                    lon_mesh, lat_mesh, mesh_values,
                    cmap=cmap, vmin=vmin, vmax=vmax, transform=projection, 
                    **kwargs_pl)
        else:
            raise ValueError("Plot type does not exist!")

        if bar:
            label = dmap.name if label is None else label
            cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                                label=label, shrink=0.8, ax=ax, **kwargs_cb)

            if bar =='discrete':
                cbar.ax.set_xticklabels(normticks[:-1]+1)

        if title is not None:
            # y_title = 1.1
            ax.set_title(title)

        return {"ax": ax, 'fig': fig, "projection": projection}


class AnomalyDataset(BaseDataset):
    """Anomaly Dataset.

    Parameters:
    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        Default: ['1997-01-01', '2019-01-01'],
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'gaussian',
    start_month: str
        Default: 'Jan'
    end_month: str
        Default: 'Dec'
    lsm: bool
        Default:False
    climatology: str
        Specified climatology the anomalies are computed over. Default: "dayofyear"
    **kwargs
    """

    def __init__(self, data_nc=None, load_nc=None,
                 var_name=None, time_range=None,
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 grid_step=1, grid_type='gaussian',
                 month_range=None,
                 lsm=False, climatology="dayofyear", **kwargs):

        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range,
                         lsm=lsm, **kwargs)

        # compute anomalies if not given in nc file
        if "anomalies" in self.vars:
            print("Anomalies are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute anomalies.")
        else:
            print(f"Compute anomalies for variable {self.var_name}.")
            da = self.ds[self.var_name]
            da = self.compute_anomalies(da, group=climatology)
            da.attrs = {"var_name": self.var_name}
            self.ds['anomalies'] = da

        # set var name to "anomalies" in order to run network on anomalies
        self.var_name = 'anomalies'

    def compute_anomalies(self, dataarray, group='dayofyear'):
        """Calculate anomalies.

        Parameters:
        -----
        dataarray: xr.DataArray
            Dataarray to compute anomalies from.
        group: str
            time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'

        Return:
        -------
        anomalies: xr.dataarray
        """
        anomalies = (dataarray.groupby(f"time.{group}")
                     - dataarray.groupby(f"time.{group}").mean("time"))
        print(f'Created {group}ly anomalies!')

        return anomalies


class EvsDataset(BaseDataset):

    def __init__(self, data_nc=None, load_nc=None,
                 var_name=None, time_range=['1997-01-01', '2019-01-01'],
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 grid_step=1, grid_type='regular',
                 month_range=None,
                 lsm=False, q=0.95, min_evs=20, min_treshold=1, th_eev=15,
                 **kwargs):

        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range,
                         lsm=lsm, **kwargs)

        # Event synchronization
        self.q = q
        self.min_evs = min_evs
        self.min_treshold = min_treshold
        self.th_eev = th_eev

        # compute event synch if not given in nc file
        if "evs" in self.vars:
            print("Evs are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute event sync.")
        else:
            print(f"Compute Event synchronization for variable {self.var_name}.")
            da = self.ds[self.var_name]
            da = self.compute_event_time_series(dataarray=da,
                                                th=self.min_treshold,
                                                q=self.q,
                                                min_evs=self.min_evs,
                                                th_eev=self.th_eev)

            da.attrs = {"var_name": self.var_name}
            self.ds['evs'] = da
            self.ds
        self.vars = []
        for name, da in self.ds.data_vars.items():
            self.vars.append(name)

    def compute_event_time_series(self, dataarray=None, th=1, q=0.95, th_eev=15, min_evs=20):

        if dataarray is None:
            dataarray = self.ds[self.var_name]
        # Remove days without rain
        data_above_th = dataarray.where(dataarray > th)
        # Compute percentile data, remove all values below percentile, but with a minimum of threshold q
        print(f"Start remove values below q={q} and at least with q_value >= {th_eev} ...")
        # Gives the quanile value for each cell
        q_mask = data_above_th.quantile(q, dim='time')
        mean_val = data_above_th.mean(dim='time')
        q_median = data_above_th.quantile(0.5, dim='time')
        # Set values below quantile to 0
        data_above_quantile = xr.where(data_above_th > q_mask[:], data_above_th, np.nan)
        # Set values to 0 that have not at least the value th_eev
        data_above_quantile = xr.where(data_above_quantile > th_eev, data_above_quantile, np.nan)
        # Remove cells with less than min_ev events.
        print(f"Remove cells without min number of events: {min_evs}")
        num_non_nan_occurence = data_above_quantile.count(dim='time')
        # Get relative amount of q rainfall to total yearly rainfall
        rel_frac_q_map = data_above_quantile.sum(dim='time') / dataarray.sum(dim='time')
        # Create mask for which cells are left out
        mask = (num_non_nan_occurence > min_evs)
        final_data = data_above_quantile.where(mask, np.nan)
        data_mask = xr.where(num_non_nan_occurence > min_evs, 1, np.nan)
        print("Now create binary event series!")
        event_series = xr.where(final_data[:] > 0, 1, 0)
        print("Done!")
        event_series = event_series.rename('evs')
        self.mask = data_mask
        self.q_mask = q_mask
        self.q_median = q_median
        self.mean_val = mean_val
        self.num_eev_map = num_non_nan_occurence
        self.rel_frac_q_map = rel_frac_q_map

        return event_series

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            print("File" + filepath + " already exists!")
            os.rename(filepath, filepath + "_backup")

        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            "q": self.q,
            "min_evs": self.min_evs,
            "min_threshold": self.min_treshold,
            "th_eev": self.th_eev,
            **self.info_dict
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)

        return None
