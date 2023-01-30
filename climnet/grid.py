"""Equidistant points on a sphere.

Fibbonachi Spiral:
https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html

Fekete points:
https://arxiv.org/pdf/0808.1202.pdf

Geodesic grid: (sec. 3.2)
https://arxiv.org/pdf/1711.05618.pdf

Review on geodesic grids:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.997&rep=rep1&type=pdf

"""
# %%
import math
from warnings import WarningMessage
import numpy as np
import matplotlib.pyplot as plt
import cartopy as ctp
 
from numpy.fft import fft
import time
from climnet.fekete import bendito, plot_spherical_voronoi
from scipy.spatial import SphericalVoronoi
import os
from scipy.spatial.transform import Rotation as Rot
import warnings
import fnmatch, sys
if fnmatch.fnmatch(sys.version, '3.7.1*'):
    import pickle5 as pickle
else:
    import pickle

class BaseGrid():
    """Base Grid

    Parameters:
    -----------

    """
    def __init__(self):
        self.grid = None
        return

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        print("Function should be overwritten by subclasses!")
        return None

    def create_grid(self):
        """Create grid."""
        print("Function should be overwritten by subclasses!")
        return None

    def cut_grid(self, lat_range, lon_range):
        """Cut the grid in lat and lon range.

        TODO: allow taking regions around the date line 

        Args:
        -----
        lat_range: list
            [min_lon, max_lon]
        lon_range: list
            [min_lon, max_lon]
        """
        if lon_range[0] > lon_range[1]:
            raise ValueError("Ranges around the date line are not yet defined.")
        else:
            print(f"Cut grid in range lat: {lat_range} and lon: {lon_range}")

        idx = np.where((self.grid['lat']>= lat_range[0]) 
                       & (self.grid['lat']<= lat_range[1]) 
                       & (self.grid['lon']>= lon_range[0]) 
                       & (self.grid['lon']<= lon_range[1]))[0] 
        cutted_grid = {'lat': self.grid['lat'][idx], 'lon': self.grid['lon'][idx]}
              
        return  cutted_grid

class RegularGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    Args:
    ----
    grid_step_lon: float 
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, grid_step_lon, grid_step_lat, grid=None):
        self.grid_step_lon = grid_step_lon
        self.grid_step_lat = grid_step_lat
        self.grid = grid
        if grid is None:
            self.create_grid()



    def create_grid(self): # lats as repeat, lons as tile
        init_lon, init_lat = regular_lon_lat_step(self.grid_step_lon, self.grid_step_lat)

        lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat)
        
        self.grid = {'lat': lat_mesh.flatten(), 'lon': lon_mesh.flatten()}

        return self.grid
        


    def get_distance_equator(self):
        """Return distance between points at the equator."""
        d_lon = degree2distance_equator(self.grid_step_lon, radius=6371)
        return d_lon

class GaussianGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    Args:
    ----
    grid_step_lon: float 
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, grid_step_lon, grid_step_lat, grid=None):
        self.grid_step_lon = grid_step_lon
        self.grid_step_lat = grid_step_lat
        self.grid = grid
        if grid is None:
            self.create_grid()



    def create_grid(self):
        init_lat = np.arange(-89.5, 90.5, self.grid_step_lat)
        init_lon = np.arange(-179.5, 180.5, self.grid_step_lon)

        lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat) # lats as repeat, lons as tile
        
        self.grid = {'lat': lat_mesh.flatten(), 'lon': lon_mesh.flatten()}

        return self.grid
        


    def get_distance_equator(self):
        """Return distance between points at the equator."""
        d_lon = degree2distance_equator(self.grid_step_lon, radius=6371)
        return d_lon



class FibonacciGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict (or 'old' makes old version of fib grid, 'maxmin' to maximize min. min-distance)

        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self, distance_between_points, grid=None, epsilon = 0.36, save = True):
        self.distance = distance_between_points
        self.num_points = self.get_num_points()
        self.epsilon = None
        self.grid = None
        if grid is None: # maxavg is standard
            self.create_grid(self.num_points, epsilon, save = save)
            self.epsilon = epsilon
        elif grid == 'old':
            self.create_grid('old')
        elif grid == 'maxmin':
            eps = maxmin_epsilon(self.num_points)
            self.create_grid(self.num_points, eps, save = save)
            self.epsilon = eps
        else:
            self.grid = grid
        self.reduced_grid = None
    

    def create_grid(self, num_points = 1, epsilon = 0.36, save =True):
        if num_points == 'old':
            """Create Fibonacci grid."""
            print(f'Create fibonacci grid with {self.num_points} points.')
            cartesian_grid = self.fibonacci_sphere(self.num_points)
            lon, lat = cartesian2spherical(cartesian_grid[:,0],
                                        cartesian_grid[:,1],
                                        cartesian_grid[:,2])
        else:
            filepath = f'fibonaccigrid_{num_points}_{epsilon}.p'
            if os.path.exists(filepath):
                print(f'\nLoad Fibonacci grid with {self.num_points} points and epsilon = {epsilon}.')
                with open(filepath, 'rb') as fp:
                    self.grid = pickle.load(fp)
                    return self.grid
            else:
                print(f'\nCreate refined fibonacci grid with {num_points} points and epsilon = {epsilon}.')
                goldenRatio = (1 + 5**0.5)/2
                i = np.arange(0, num_points) 
                theta = 2 *np.pi * i / goldenRatio
                phi = np.arccos(1 - 2*(i+epsilon)/(num_points-1+2*epsilon))
                x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
                lon, lat = cartesian2spherical(x,y,z)
                self.grid = {'lon': lon, 'lat': lat}
                if save:
                    with open(filepath, 'wb') as fp:
                        pickle.dump(self.grid, fp, protocol=pickle.HIGHEST_PROTOCOL)
        self.grid = {'lon': lon, 'lat': lat}
        return self.grid

    def fibonacci_sphere(self, num_points=1):
        """Creates the fibonacci sphere points on a unit sphere.
        Code inspired by:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)


    def fibonacci_refined(self, num_points = 1, epsilon = 0.36): # epsilon = 0.36 for optimal average distance
            
        print(f'Create refined fibonacci grid with {self.num_points} points and epsilon={epsilon}.')

        goldenRatio = (1 + 5**0.5)/2
        i = np.arange(0, num_points) 
        theta = 2 *np.pi * i / goldenRatio
        phi = np.arccos(1 - 2*(i+epsilon)/(num_points-1+2*epsilon))
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        lon, lat = cartesian2spherical(x,y,z)
        self.grid = {'lon': lon, 'lat': lat}
        return self.grid

    def get_num_points(self):
        """Relationship between distance and num of points of fibonacci sphere.

        num_points = a*distance**k
        """
        # obtained by log-log fit 
        k = -2.01155176
        a = np.exp(20.0165958)

        return int(a * self.distance**k)


    def fit_numPoints_distance(self):
        """Fit functional relationship between next-nearest-neighbor distance of
           fibonacci points and number of points."""
        num_points = np.linspace(200, 10000, 20, dtype=int)
        main_distance = []
        for n in num_points:
            points = self.fibonacci_sphere(n)
            lon, lat = cartesian2spherical(points[:,0], points[:,1], points[:,2])

            distance = neighbor_distance(lon, lat)
            hist, bin_edge = np.histogram(distance.flatten(), bins=100)

            main_distance.append(bin_edge[np.argmax(hist)])

        # fit function
        logx = np.log(main_distance)
        logy = np.log(num_points)
        coeffs = np.polyfit(logx, logy, deg=1)

        def func(x):
            return np.exp(coeffs[1]) * x**(coeffs[0])

        dist_array = np.linspace(100, 1400, 20)
        y_fit = func(dist_array)

        # Plot fit and data
        fig, ax = plt.subplots()
        ax.plot(main_distance, num_points)
        ax.plot(dist_array, y_fit )
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        return coeffs

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return  self.distance

    
    def keep_original_points(self, orig_grid, regular = True):
        if self.grid is None:
            self.create_grid()
        if regular:
            new_lon, new_lat, dists = [], [], []
            lons = np.sort(np.unique(orig_grid['lon']))
            lats = np.sort(np.unique(orig_grid['lat']))
            for i in range(len(self.grid['lat'])):
                lo = self.grid['lon'][i]
                la = self.grid['lat'][i]
                pm_lon = np.array([(lons[j]-lo)*(lons[j+1]-lo) for j in range(len(lons)-1)])
                pm_lat = np.array([(lats[j]-la)*(lats[j+1]-la) for j in range(len(lats)-1)])
                if np.where(pm_lon<0)[0].shape[0] == 0:
                    rel_lon = [lons[0], lons[-1]]
                else:
                    lon_idx = np.where(pm_lon<0)[0][0]
                    rel_lon = [lons[lon_idx], lons[lon_idx+1]]
                if np.where(pm_lat<0)[0].shape[0] == 0:
                    rel_lat = [lats[0], lats[-1]]
                else:
                    lat_idx = np.where(pm_lat<0)[0][0]
                    rel_lat = [lats[lat_idx], lats[lat_idx+1]]
                min_dist = 99999
                for l1 in rel_lon:
                    for l2 in rel_lat:
                        this_dist = gdistance((l2,l1),(la,lo))
                        if this_dist < min_dist:
                            min_lon, min_lat = l1, l2
                            min_dist = this_dist
                new_lon.append(min_lon)
                new_lat.append(min_lat)
                dists.append(min_dist)
            self.reduced_grid = {'lon': np.array(new_lon), 'lat': np.array(new_lat)}
            return np.array(dists)
        else:
            raise KeyError('Only regular grids!')

    def min_dists(self, grid2 = None):
        if grid2 is None:
            lon1, lon2 = self.grid['lon'], self.grid['lon']
            lat1, lat2 = self.grid['lat'], self.grid['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range((len(lon1))):
                    if i < j:
                        d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
                    elif i>j:
                        d[i,j] = d[j,i]
            return d.min(axis=1)
        else: # min dist from self.grid point to other grid
            lon1, lon2 = self.grid['lon'], grid2['lon']
            lat1, lat2 = self.grid['lat'], grid2['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range(len(lon2)):
                    d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
            return d.min(axis=1)


class FeketeGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict (or 'old' makes old version of fib grid, 'maxmin' to maximize min. min-distance)

        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self, num_points, num_iter = 1000, grid=None, save = True):
        self.distance = get_distance_from_num_points(num_points)
        self.num_points = num_points
        self.num_iter = num_iter
        self.epsilon = None
        self.grid = grid
        if grid is None: # maxavg is standard
            self.create_grid(num_points, num_iter, save = save)
        self.reduced_grid = None
    

    def create_grid(self, num_points = 1, num_iter = 1000, save = True):
        filepath = f'feketegrid_{num_points}_{num_iter}.p'
        if os.path.exists(filepath):
            print(f'\nLoad Fekete grid with {self.num_points} points after {num_iter} iterations.')
            with open(filepath, 'rb') as fp:
                self.grid, self.dq = pickle.load(fp)
                return self.grid
        else:   
            print(f'\nCreate Fekete grid with {self.num_points} points after {num_iter} iterations.')
            X, self.dq = bendito(N=num_points, maxiter=num_iter)
            lon, lat = cartesian2spherical(X[:,0], X[:,1], X[:,2])
            self.grid = {'lon': lon, 'lat': lat}
            if save:
                with open(filepath, 'wb') as fp:
                    pickle.dump((self.grid,self.dq), fp, protocol=pickle.HIGHEST_PROTOCOL)
            return self.grid

    def nudge_grid(self, n_iter = 1, step = 0.01): # a 100th of a grid_step
        if self.reduced_grid is None:
            raise KeyError('First call keep_original_points')
        leng = len(self.grid['lon'])
        delta = 2 * np.pi * step * self.distance / 6371
        regx, regy, regz = spherical2cartesian(self.reduced_grid['lon'], self.reduced_grid['lat'])
        for iter in range(n_iter):
            perm =np.random.permutation(leng)
            for i in range(leng):
                i = perm[i]
                x,y,z = spherical2cartesian(self.grid['lon'], self.grid['lat'])
                r = np.array([x[i],y[i],z[i]])
                vec2 = np.array([regx[i] - x[i], regy[i] -y[i], regz[i]-z[i]])
                vec2 = vec2 - np.dot(vec2, r) * r
                rot_axis = np.cross(r, vec2)
                rot = Rot.from_rotvec(rot_axis * delta / np.linalg.norm(rot_axis))
                new_grid_cart = rot.as_matrix() @ np.array([x, y, z])
                new_lon, new_lat = cartesian2spherical(new_grid_cart[0,:], new_grid_cart[1,:], new_grid_cart[2,:])
                self.grid = {'lon': new_lon, 'lat': new_lat}


    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return  self.distance

    
    def keep_original_points(self, orig_grid, regular = True):
        if self.grid is None:
            self.create_grid()
        if regular:
            new_lon, used_dists, delete, dists, possible_coords, new_coords = [], [], [], [], [], []
            lons = np.sort(np.unique(orig_grid['lon']))
            lats = np.sort(np.unique(orig_grid['lat']))
            for i in range(len(self.grid['lat'])):
                lo = self.grid['lon'][i]
                la = self.grid['lat'][i]
                pm_lon = np.array([(lons[j]-lo)*(lons[j+1]-lo) for j in range(len(lons)-1)])
                pm_lat = np.array([(lats[j]-la)*(lats[j+1]-la) for j in range(len(lats)-1)])
                if np.where(pm_lon<0)[0].shape[0] == 0:
                    rel_lon = [lons[0], lons[-1]]
                else:
                    lon_idx = np.where(pm_lon<0)[0][0]
                    rel_lon = [lons[lon_idx], lons[lon_idx+1]]
                if np.where(pm_lat<0)[0].shape[0] == 0:
                    rel_lat = [lats[0], lats[-1]]
                else:
                    lat_idx = np.where(pm_lat<0)[0][0]
                    rel_lat = [lats[lat_idx], lats[lat_idx+1]]
                these_dists = np.array([gdistance((l2,l1),(la,lo)) for l1 in rel_lon for l2 in rel_lat])
                these_coords = np.array([(l1,l2) for l1 in rel_lon for l2 in rel_lat])
                prio = np.argsort(these_dists)
                dists.append(these_dists[prio])
                possible_coords.append(these_coords[prio])

            for idx in np.argsort(np.array(dists)[:,0]): # choose the nearest unused neighbor
                i = 0
                while i < 4 and np.any([np.all(possible_coords[idx][i,:] == coord) for coord in new_coords]):
                    if i == 3:
                        delete.append(idx)
                        warnings.warn(f'No neighbors left for  {self.grid["lon"][idx], self.grid["lat"][idx]}. Removing this point.')
                    i += 1
                if i < 4:
                    new_coords.append(possible_coords[idx][i,:])
                    used_dists.append(dists[idx][i])
                # ids = map(id, new_coords)
            dists2 = np.delete(dists, delete, 0)
            new_coords = np.array(new_coords)[np.argsort(np.argsort(np.array(dists2)[:,0]))] # inverse permutation
            self.reduced_grid = {'lon': np.array(new_coords)[:,0], 'lat': np.array(new_coords)[:,1]}
            self.grid['lon'] = np.delete(self.grid['lon'], delete,0)
            self.grid['lat'] = np.delete(self.grid['lat'], delete,0)
            return used_dists
        else:
            raise KeyError('Only regular grids!')

    def min_dists(self, grid2 = None):
        if grid2 is None:
            lon1, lon2 = self.grid['lon'], self.grid['lon']
            lat1, lat2 = self.grid['lat'], self.grid['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range((len(lon1))):
                    if i < j:
                        d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
                    elif i>j:
                        d[i,j] = d[j,i]
            return d.min(axis=1)
        else: # min dist from self.grid point to other grid
            lon1, lon2 = self.grid['lon'], grid2['lon']
            lat1, lat2 = self.grid['lat'], grid2['lat']
            d = 9999 * np.ones((len(lon1),len(lon2)))
            for i in range(len(lon1)):
                for j in range(len(lon2)):
                    d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
            return d.min(axis=1)


def get_distance_from_num_points(num_points):
    k = 1/2.01155176
    a = np.exp(20.0165958)
    return (a/num_points)**k

def get_num_points(dist):
        """Relationship between distance and num of points of fibonacci sphere.

        num_points = a*distance**k
        """
        # obtained by log-log fit 
        k = -2.01155176
        a = np.exp(20.0165958)
        return int(a * dist**k)

def maxmin_epsilon(num_points):
    if num_points >= 600000:
        epsilon = 214
    elif num_points>= 400000:
        epsilon = 75
    elif num_points>= 11000:
        epsilon = 27
    elif num_points>= 890:
        epsilon = 10
    elif num_points>= 177:
        epsilon = 3.33
    elif num_points>= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33
    return epsilon

def regular_lon_lat(num_lon, num_lat): # creates regular grid with borders half the distance of one step at each border
    lon = np.linspace(-180+360/(2*num_lon),180-360/(2*num_lon),num_lon)
    lat = np.linspace(-90 + 180/(2*num_lat), 90 - 180/(2*num_lat), num_lat)
    return lon, lat

def regular_lon_lat_step(lon_step, lat_step): # creates next best finer grid symmetric to borders
    num_lon = int(np.ceil(360/lon_step))
    lon_border = (360 - (num_lon-1) * lon_step)/2
    num_lat = int(np.ceil(180/lat_step))
    lat_border = (180 - (num_lat-1) * lat_step)/2
    lon = np.linspace(-180 + lon_border, 180 - lon_border, num_lon)
    lat = np.linspace(-90 + lat_border, 90 - lat_border, num_lat)
    return lon, lat

def cartesian2spherical(x, y, z):
    """Cartesian coordinates to lon and lat.
    
    Args:
    -----
    x: float or np.ndarray
    y: float or np.ndarray
    z: float or np.ndarray
    """
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lon, lat

def spherical2cartesian(lon, lat):
    lon = lon * 2 *np.pi / 360
    lat = lat * np.pi / 180
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return x, y, z

def gdistance(pt1, pt2, radius=6371.009):
    lon1, lat1 = pt1[1], pt1[0]
    lon2, lat2 = pt2[1], pt2[0]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

@np.vectorize
def haversine(lon1, lat1, lon2, lat2, radius=1):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

def degree2distance_equator(grid_step, radius=6371):
    """Get distance between grid_step in km"""
    distance = haversine(0,0, grid_step, 0, radius=radius)
    return distance

def distance2degree_equator(distance, radius = 6371):
    return distance * 360/(2 * np.pi * radius)


def neighbor_distance(lon, lat, radius=6371):
    """Distance between next-nearest neighbor points on a sphere.
    Args:
    -----
    lon: np.ndarray
        longitudes of the grid points
    lat: np.ndarray
        latitude values of the grid points

    Return:
    -------
    Array of next-nearest neighbor points
    """
    distances = []
    for i in range(len(lon)):
        d = haversine(lon[i], lat[i], lon, lat, radius)
        neighbor_d = np.sort(d)
        distances.append(neighbor_d[1:2])

    return np.array(distances)

def min_dists(grid1, grid2 = None):
    if grid2 is None:
        lon1, lon2 = grid1['lon'], grid1['lon']
        lat1, lat2 = grid1['lat'], grid1['lat']
        d = 9999 * np.ones((len(lon1),len(lon2)))
        for i in range(len(lon1)):
            for j in range((len(lon1))):
                if i < j:
                    d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
                elif i>j:
                    d[i,j] = d[j,i]
        return d.min(axis=1)
    else: # min dist from self.grid point to other grid
        lon1, lon2 = grid1['lon'], grid2['lon']
        lat1, lat2 = grid1['lat'], grid2['lat']
        d = 9999 * np.ones((len(lon1),len(lon2)))
        for i in range(len(lon1)):
            for j in range(len(lon2)):
                d[i,j] = gdistance((lat1[i],lon1[i]),(lat2[j],lon2[j]))
        return d.min(axis=1)


# %%
if __name__ == "__main__":
    # Test
    grid_step = 5
    num_lon = int(np.ceil(360/grid_step))
    num_lat = int(np.ceil(180/grid_step))
    dist_equator = degree2distance_equator(grid_step)
    num_points = get_num_points(dist_equator)

    # %%
    reg_lon, reg_lat = regular_lon_lat_step(grid_step, grid_step)
    print(' Adjusted lon step: ', reg_lon[1]-reg_lon[0], '\n Adjusted lat step: ', reg_lat[1]-reg_lat[0])
    reg_pregrid = {'lon': reg_lon, 'lat': reg_lat}

    # %%
    Fek = FeketeGrid(num_points)
    dists = Fek.keep_original_points(reg_pregrid)
    proj = ctp.crs.Mollweide()
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)
    ax.stock_img()
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    for i in range(len(Fek.grid['lon'])):
        ax.plot([Fek.grid['lon'][i],Fek.reduced_grid['lon'][i]], [Fek.grid['lat'][i],Fek.reduced_grid['lat'][i]], color = 'black', linestyle='-', transform=ctp.crs.PlateCarree())
    
    
    # %%
    plt.hist(dists)
    print('mean dists within', np.mean(dists))
    # %%
    Fek.nudge_grid(step = 0.005)
    dists2 = Fek.keep_original_points(reg_pregrid)
    plt.hist(dists2)
    print('mean dists within:', np.mean(dists2))

    # %%
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)
    ax.stock_img()
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    for i in range(len(Fek.grid['lon'])):
        ax.plot([Fek.grid['lon'][i],Fek.reduced_grid['lon'][i]], [Fek.grid['lat'][i],Fek.reduced_grid['lat'][i]], color = 'black', linestyle='-', transform=ctp.crs.PlateCarree())

    # %%
    # reg_lon2, reg_lat2 = regular_lon_lat(num_lon, num_lat)
    # print(' Adjusted lon step: ', reg_lon2[1]-reg_lon2[0], '\n Adjusted lat step: ', reg_lat2[1]-reg_lat2[0])
    # reg_pregrid2 = {'lon': reg_lon2, 'lat': reg_lat2}
    # reg_pregrid2['lon'], reg_pregrid['lon']
    # %%
    Fib_old = FibonacciGrid(dist_equator, 'old')
    Fib_maxavg = FibonacciGrid(dist_equator)
    Fib_maxmindist = FibonacciGrid(dist_equator, grid = 'maxmin')
    lon, lat = Fib_old.grid['lon'], Fib_old.grid['lat']
    lon2, lat2 = Fib_maxavg.grid['lon'], Fib_maxavg.grid['lat']
    lon3, lat3 = Fib_maxmindist.grid['lon'], Fib_maxmindist.grid['lat']
    # %%
    proj = ctp.crs.EqualEarth()
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)
    ax.stock_img()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    # ax.axis('off')
    #ax.scatter(Fib.grid['lon'], Fib.grid['lat'], markersize = 0.7, transform = ctp.crs.PlateCarree())
    ax.plot(lon, lat, 'bo', markersize = 1,label = 'old', transform = ctp.crs.PlateCarree())
    ax.plot(lon2, lat2, 'ro', markersize = 1,label = 'refined_maxavg', transform = ctp.crs.PlateCarree())
    ax.plot(lon3, lat3, 'go', markersize = 1,label = 'refined_maxmin', transform = ctp.crs.PlateCarree())
    ax.legend()

    # %%
    old_grid_distances = Fib_old.keep_original_points(reg_pregrid)
    plt.hist(old_grid_distances, bins =100)
    plt.title('Min distances between regular and old fib grid')
    plt.xlabel('Distance (km)')
    # %%
    maxavg_distances = Fib_maxavg.keep_original_points(reg_pregrid)
    plt.hist(maxavg_distances, bins =100)
    plt.title('Min distances between regular and maxavg fib grid')
    plt.xlabel('Distance (km)')
    # %%
    maxmin_distances = Fib_maxmindist.keep_original_points(reg_pregrid)
    plt.hist(maxmin_distances, bins =100)
    plt.title('Min distances between regular and maxmin fib grid')
    plt.xlabel('Distance (km)')
    
    # %%
    print(' old mean dist to reg grid: ', np.mean(old_grid_distances), '\n maxmin mean dist to reg grid: ', np.mean(maxmin_distances),
    '\n maxavg mean dist to reg grid: ', np.mean(maxavg_distances))
    cross_dists = Fib_maxavg.min_dists(Fib_old.grid)
    old_mindists = Fib_old.min_dists()
    maxmin_mindists = Fib_maxmindist.min_dists()
    # same as:
    # nnn_dists = neighbor_distance(Fib_maxavg.grid['lon'], Fib_maxavg.grid['lat'])
    maxavg_mindists = Fib_maxavg.min_dists()
    print(' old mean mindist: ', np.mean(old_mindists), '\n maxmin mean min dist: ', np.mean(maxmin_mindists),
    '\n maxavg mean mindist: ', np.mean(maxavg_mindists))

    # %%
    plt.hist(old_mindists, bins = 100)
    plt.title('Min distances within old fib grid')
    plt.xlabel('Distance (km)')
    # %%
    plt.hist(maxavg_mindists, bins = 100)
    plt.title('Min distances within maxavg fib grid')
    plt.xlabel('Distance (km)')
    # %%
    plt.hist(maxmin_mindists, bins = 100)
    plt.title('Min distances within maxmin fib grid')
    plt.xlabel('Distance (km)')    
   
    # %%
    proj = ctp.crs.Mollweide()
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)
    ax.stock_img()
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    for i in range(len(Fib_maxavg.grid['lon'])):
        ax.plot([Fib_maxavg.grid['lon'][i],Fib_maxavg.reduced_grid['lon'][i]], [Fib_maxavg.grid['lat'][i],Fib_maxavg.reduced_grid['lat'][i]], color = 'black', linestyle='-', transform=ctp.crs.PlateCarree())
    
    # %%
    # Test to compare to classical grid
    start = time.time()
    lon_mesh, lat_mesh = np.meshgrid(reg_lon, reg_lat)
    nndist_reg = neighbor_distance(lon_mesh.flatten(), lat_mesh.flatten())
    print('vectorized', time.time()-start)
    plt.hist(nndist_reg.flatten(), bins=100)

    '''
    # Does the same:
    reg_grid2 = RegularGrid(grid_step, grid_step)
    start = time.time()
    nndist_reg2 = min_dists(reg_grid2.grid)
    print('mine', time.time()-start)
    plt.hist(nndist_reg2.flatten(), bins=100)
    '''
    # %%
    nndist_reg.shape
    # %%
    X, dq = bendito(N=len(Fib_maxavg.grid['lon']))
    # %%
    fek_lat, fek_lon = cartesian2spherical(X[:,0],X[:,1],X[:,2])
    fek_grid = {'lon':fek_lon, 'lat': fek_lat}
    fek_mindists = min_dists(fek_grid)
    print('fekete mean mindists:', np.mean(fek_mindists))
    # %%
    plt.hist(fek_mindists, bins = 100)
    plt.title('Min distances within fekete grid')
    plt.xlabel('Distance (km)')

    # %%
    proj = ctp.crs.EqualEarth()
    # plate = ctp.crs.PlateCarree()
    # transf_points = plate.transform_points(ctp.crs.Geocentric(), x= X[:,0], y = X[:,1], z = X[:,2])[:,:2]
    fig, ax = plt.subplots(figsize=(9,6))
    ax = plt.axes(projection=proj)
    ax.stock_img()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
    ax.plot(fek_grid['lon'], fek_grid['lat'], 'bo', markersize = 5,label = 'fekete', transform = ctp.crs.PlateCarree())
    ax.plot(lon2, lat2, 'ro', markersize = 1,label = 'refined_maxavg', transform = ctp.crs.PlateCarree())
    #ax.plot(transf_points[:,0], transf_points[:,1], 'bo', markersize = 5,label = 'fekete', transform = plate)
    ax.legend()

    # %%
    try:
        a = fek_grid['lol']
    except:
        a = None
    print(a)
# %%
