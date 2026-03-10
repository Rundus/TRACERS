from ...lib.ACE.ace_data_loading import *
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import pandas as pd
from ...lib.general.misc_functions import determine_datetime_type




def get_characteristic_energies(l2_data,trange=None,return_times=None):
    """
    Required input:
    - ACE data dictionary generated from ace_data_loading routines
    Optional input:
    - time range you wish to examine if you want to only look at a subset of the data you have loaded.
      trange = ['2025-09-13/12:13:43','2025-09-13/12:13:50'], for example
    """
    flux = l2_data['def'] # T x 49 x 21
    datetimes = np.array(l2_data['DT'])
    energy = l2_data['energy']

    flux[flux < 0] = 0 # replacing negative flux values with zero to get better outcome

    if trange is not None:
        t0 = dt.datetime.strptime(trange[0],determine_datetime_type(trange[0])).replace(tzinfo=dt.timezone.utc)
        tf = dt.datetime.strptime(trange[1],determine_datetime_type(trange[1])).replace(tzinfo=dt.timezone.utc)

        mask = (datetimes >= t0)&(datetimes <= tf)
        flux = flux[mask,:,:]
        datetimes = datetimes[mask]
    
    efull = np.repeat(energy[np.newaxis,:],len(datetimes),axis=0) # T x E 
    avg_def = np.nanmean(flux,axis=2) # averaging flux over the anodes
    characteristic_energy = np.sum(efull*avg_def,axis=1)/np.sum(avg_def,axis=1)
    if return_times is None or return_times is False:
        return characteristic_energy
    else:
        return datetimes, characteristic_energy







def calculate_energy_flux(l3_data,replace_negatives=None,angle_subset=None):
    delta_e = np.zeros(49)
    energies = l3_data['energy']
    
    delta_e[1:48] = abs((energies[2:]-energies[0:47])/2.)
    delta_e[0] = delta_e[1]*energies[0]/energies[1]
    delta_e[48] = delta_e[47]*energies[48]/energies[47]
    
    denergies = np.zeros((49,18))
    for a in range(18):
        denergies[:,a] = delta_e

    theta_vals = 90. - np.arange(18)*10 - 5
    thetas = np.zeros((49,18))
    pas = np.zeros((49,18))
    for i in range(49):
        thetas[i,:] = theta_vals
        pas[i,:] = l3_data['pitch_angle']
    dthetas = 10*np.ones((49,18))
    
    phis = np.zeros((49,18))
    dphis = 360.*np.ones((49,18))
    domega = 2*np.deg2rad(dphis)*np.cos(np.deg2rad(thetas))*np.sin(0.5*np.deg2rad(dthetas))
    
    energy_fluxes = np.zeros((3,len(l3_data['UTC'])))
    omni_energy_fluxes = np.zeros(len(l3_data['UTC']))
    for t in range(len(l3_data['UTC'])):
        dist =l3_data['def'][t,:,:]
        if replace_negatives is True:
            dist[dist < 0] = 0

        if angle_subset is None:
            omni = np.sum(dist*domega*np.cos(np.deg2rad(pas)),axis=1)
        else:
            val = dist*domega*np.cos(np.deg2rad(pas))
            val_subset = val[:,angle_subset]
            omni = np.sum(val_subset,axis=1)
            
        omni_erg = np.sum((1.6022e-12)*delta_e*omni) 
        omni_energy_fluxes[t] = omni_erg                      

    return omni_energy_fluxes







    
    
def get_velocity_components(l3_data):
    """
    Returns arrays of perpendicular and parallel velocities for each energy/pitch_angle pair.
    """
    pitch_angles = l3_data['pitch_angle']
    energy = l3_data['energy']

    e_matrix = np.zeros((49,18))
    for i in range(18):
        e_matrix[:,i] = energy
    pa_matrix = np.zeros((49,18))
    for j in range(49):
        pa_matrix[j,:] = pitch_angles
    
    v_par = np.sqrt(2*e_matrix*1.6e-19/9.1e-31)*np.cos(np.deg2rad(pa_matrix))    
    v_perp = np.sqrt(2*e_matrix*1.6e-19/9.1e-31)*np.sin(np.deg2rad(pa_matrix))    
        
    return v_perp, v_par


def plot_epad_snapshot(l3_data,ax=None,units=None,cmap=None,n_levels=None):
    
    if ax is None:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
    if cmap is None:
        cmap = cc.cm['CET_R3_r']
    else:
        cmap = plt.get_cmap(cmap)
    cmap = copy.copy(cmap)
    cmap.set_under('black')

    if n_levels is None:
        n_levels=20
    
    if units is None or units.lower() == 'def':
        particle_data = l3_data['def']
        zlabel = 'Differential Energy Flux\n'+r'eV/[eV s sr cm$^{2}$]'
    if units.lower() == 'counts':
        particle_data = l3_data['counts']
        zlabel = 'Counts'
    if units.lower() == 'df':
        pd = l3_data['def']
        # rest mass of electron m = E/c2
        mass = 5.68566e-16 # eV/(cm2/s2)
        e_matrix = np.zeros((49,18))
        for i in range(18):
            e_matrix[:,i] = l3_data['energy']
        df_conversion = 0.5*mass**2/(e_matrix**2) # [eV2/(cm2/s2)]/eV2
        particle_data = l3_data['def']*df_conversion*(100**6) # s3/m6
        zlabel = 'Distribution Function\n'+r'[s$^3$/m$^6$]'
        

    v_perp, v_par = get_velocity_components(l3_data)

    time_sum = np.nansum(particle_data,axis=0)
    time_sum = np.ma.masked_where(time_sum <= 0, time_sum)
    
    lev_exp = np.linspace(np.floor(np.log10(time_sum.min())),
                       np.ceil(np.log10(time_sum.max())),n_levels)
    levs = np.power(10, lev_exp)
    
    cbar_ticks = np.power(10, np.arange(lev_exp[0], lev_exp[-1]))
    
    cm = ax.contourf(v_par, v_perp, time_sum, levs, norm=colors.LogNorm(), cmap=cmap)
    ax.set_xlim([-6.3e7, 6.3e7])
    ax.set_ylim([0,6.3e7])
    ax.set_xlabel(r'$v_{\parallel}$ [m/s]')
    ax.set_ylabel(r'$v_{\perp}$ [m/s]')
    cbax = ax.inset_axes([1.03,0,0.08,1],transform=ax.transAxes)
    plt.colorbar(cm,cax=cbax,ticks=cbar_ticks,label=zlabel)
    return ax
    
