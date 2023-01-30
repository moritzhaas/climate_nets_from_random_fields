
# %%
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
from climnet.myutils import *
import time
from climnet.grid import FeketeGrid, regular_lon_lat
from sklearn.gaussian_process.kernels import Matern
start_time = time.time()




dists = myload(base_path + f'grids/fekete_dists_npoints_{num_points}.txt')
min_dists = dists
min_dists[np.eye(len(dists),dtype = bool)] = 999999
min_dists = min_dists.min(axis=0)
plt.hist(min_dists,bins = 20)
plt.title(f'Min dists to next pt in Fekete grid with {num_points} pts')
plt.savefig(base_path+f'mindists_fekete_{num_points}.pdf',dpi=150)
# %%
from climnet.grid  import FibonacciGrid
earth_radius = 6371.009
dist_equator = gdistance((0,0),(0,2.5))
Fib_maxavg = FibonacciGrid(dist_equator)
lon3, lat3 = Fib_maxavg.grid['lon'], Fib_maxavg.grid['lat']
# %%
dists3 = np.zeros((len(lon3), len(lon3)))
for i in range(len(lon3)):
    for j in range(i):
        dists3[i,j] = gdistance((lat3[i], lon3[i]), (lat3[j],lon3[j]), radius=1)
        dists3[j,i] = dists3[i,j]

# %%
dists3 = myload(base_path+f'grids/fib_maxavg_dists_npoints_{num_points}.txt')
min_dists3 = dists3
min_dists3[np.eye(len(dists3),dtype = bool)] = 999999
min_dists3 = min_dists3.min(axis=0)
plt.hist(min_dists3,bins = 20)
plt.title(f'Min dists to next pt in Fibonacci grid with {num_points} pts')
#plt.yscale('log')
plt.savefig(base_path+f'mindists_fibonacci_{num_points}.pdf',dpi=150)
# %%
num_points = gridstep_to_numpoints(2.5)
mysave(base_path+'grids/', f'fib_maxavg_dists_npoints_{num_points}.txt', dists3)
mysave(base_path+'grids/', f'fib_maxavg_grid_npoints_{num_points}.txt', Fib_maxavg.grid)