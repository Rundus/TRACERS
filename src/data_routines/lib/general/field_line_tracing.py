import PyGeopack as gp
import matplotlib.pyplot as plt
import numpy as np

def plot_geopack_trace(r, yyyymmdd, hh, az=70, elev=35,
                       ax=None, cmap='viridis',model='TS05',return_full=False,
                       labels=False, legend=False):
    '''
    Given a series of position vectors, trace the field line of each position and plot it.
    Defaults to using a Tsyganenko-Sitnov 05 magnetic field model.

    Inputs:
     r: position of spacecraft in km in SM frame. Dimensions N x 3 where N is the number of field lines
     yyyymmdd: Year - month - day to calculate the magnetic field model at
     hh: Decimal hours to calculate the magnetic field model at
     az: Camera azimuth
     elev: Camera elevation
     cmap: Colormap to color sequential lines as
     model: Magnetic field model to use. Options: 'T89'|'T96'|'T01'|'TS05'
     return_full: If true, return (axis, trace) where trace is trace object.
                  Useful for getting the footprints and equatorial local time
     labels: Associate lines with labels if true. List of strings of length N
     legend: Place legend on axes. 

    '''
    RE = 6378.0; # Earth radius in km
    T = gp.TraceField(r[:,0]/RE , r[:,1]/RE, r[:,2]/RE,
                      yyyymmdd,hh,Model=model,CoordIn='SM',CoordOut='GSM')

    
    if not ax:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        
    # Get the colormap and generate a list of colors
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, r.shape[0])]

    ax.view_init(elev=elev, azim=az)
    ax.set_prop_cycle('color', colors)

    for i in range(r.shape[0]):
        pl,=ax.plot(T.xgsm[i,:], T.ygsm[i,:], T.zgsm[i,:])
        if labels:
            pl.set_label(labels[i])

    ax.set_xlabel('X ($R_E$, GSM)')
    ax.set_ylabel('Y ($R_E$, GSM)')
    ax.set_zlabel('Z ($R_E$, GSM)')

    if legend:
        ax.legend(loc='upper right')

    if not return_full:
        return ax
    else:
        return ax, T
