import aacgmv2._aacgmv2 as c_aacgmv2
import aacgmv2
import cartopy
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import sys
from lib.general.ead_file_loading import *



def geographic_plot(t0, tf, sc, ax=None, nightshade=None, hemisphere=None, cmap=None, orbit_path=None):
    """
    Generates an axis object with continent outlines and day/night coloring for a given TRACERS observation.
    
    Inputs: 
    t0 = start time of observation (string)
    tf = end time of observation (string)
    sc = spacecraft number ('1' or '2')

    Optional:
    ax = axis object 
    nightshade = True or False (true to add nightshade to plot)
    add_aurora = True or False (true to add aurora signatures to plot, default is False)
    orbit_path = True or False (default True; plots path of TRACERS satellite)
    """
    
    ead = EADload(t0,tf,sc)
    ead_data = ead.read_data(params=['lat_geo','lon_geo','lat_geod','lon_geod','alt_geod','mlat','mlt','r_sm'])
    n_times = len(ead_data['DT'])
    middle_idx = int(np.floor(n_times/2))
    mid_time = ead_data['DT'][middle_idx]
        
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        if hemisphere is None or hemisphere.lower() == 'north':
            ax = fig.add_subplot(111, projection=ccrs.Orthographic(0, 90))
        else:
            ax = fig.add_subplot(111, projection=ccrs.Orthographic(0,-90))
    
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'), zorder=0)
    
    if nightshade is True:
        ax.add_feature(Nightshade(mid_time),alpha=0.2)
    
    #   lon_min, lon_max, lat_min, lat_max
    if hemisphere is None or hemisphere.lower() == 'north':
        ax.set_extent((0,360,30,90),ccrs.PlateCarree())
    else:
        ax.set_extent((0,360,-90,-30),ccrs.PlateCarree())
    ax.gridlines()
    

    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    if cmap is None:
        cmap = 'cividis'

    if orbit_path is None or orbit_path is True:
        cm = ax.scatter(ead_data[f'ts{sc}_ead_lon_geo'],ead_data[f'ts{sc}_ead_lat_geo'],c=ead_data['UTC'],\
                        cmap=cmap,zorder=2,marker='.',s=9, transform=ccrs.PlateCarree())
    
    
    return ax










    