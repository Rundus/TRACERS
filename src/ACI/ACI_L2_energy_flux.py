# --- ACI_L2_energy_flux ---
# Calculates the energy flux from the ACE detector
import spaceToolsLib as stl
from glob import glob
import numpy as np
from scipy.stats import energy_distance
from scipy.integrate import simpson
from tqdm import tqdm
from copy import deepcopy
import datetime as dt
from src.data_paths import DataPaths



# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---
Emin_GR2001 = 2000 # For Robinson 1987, he suggests only using electron energies above 500 eV since below this tends to ionize the F-Region

# Calculate the rough pitch angle to integrate up to
lat = [78.4]*2
long = [-144.7]*2
alt = [600, 200]
times = [dt.datetime(2024,9,27,22,18)]*2
B_ENU = stl.CHAOS(lat,long,alt,times)
B = [np.linalg.norm(vec) for vec in B_ENU]
pitch_target_val = np.degrees(np.arcsin(np.sqrt(B[0]/B[1])))

def ACI_L2_to_energy_flux():

    # Which data
    year='2025'
    month = '09'
    day = '27'
    SC = '2'

    # load the ACI data
    data_files = glob(f'{DataPaths.TRACERS_data_folder}/ACI/' +f'{year}/{month}/{day}/ts{SC}/*aci_ipd_*')
    data_dict_ACI = stl.loadDictFromFile(data_files[0])

    # --- --- --- --- --- --- --- --- ---
    # --- Calculate the Energy Flux ---
    # --- --- --- --- --- --- --- --- ---

    # Get the detector info
    epoch = data_dict_ACI['Epoch'][0]
    Energy = data_dict_ACI['ts2_l2_aci_energy'][0]
    pitch_angle = data_dict_ACI['ts2_l2_aci_tscs_anode_angle'][0]
    diffEFlux = data_dict_ACI['ts2_l2_aci_tscs_def'][0]

    # Determine the index of energy values to integrate over for the robisnon formulae
    engy_idx_GR2001 = np.abs(Energy - Emin_GR2001).argmin()

    # Determine the index of the pitch angle values to integrate over
    ptch_idx_min = np.abs(pitch_angle - -1*pitch_target_val).argmin()
    ptch_idx_max = np.abs(pitch_angle - pitch_target_val).argmin()
    ptch_idx_0 = np.abs(pitch_angle - 0).argmin()

    # loop through each time and calculate the energy flux
    energy_flux_earthward = np.zeros(shape=(len(epoch)))
    energy_flux_earthward_GR2001 = np.zeros(shape=(len(epoch)))
    characteristic_energy = np.zeros(shape=(len(epoch)))

    for tmeIdx in tqdm(range(len(epoch))):

        # Get the sweep data
        sweepData = diffEFlux[tmeIdx]
        sweepData[sweepData<0] = 0 # remove any possible negative values

        # Integrate over -90 to -0 polar, 0 to pi azimuth
        pitches = np.radians(pitch_angle[ptch_idx_min:ptch_idx_0+1]).round(decimals=3)
        factor = np.sin(pitches)*np.cos(pitches)
        J_E_neg = np.pi*simpson(y=sweepData[:,ptch_idx_min:ptch_idx_0+1]*factor.round(decimals=3), x=pitches)

        # Integrate over 0 to 90, 0 to pi azimuth
        pitches = np.radians(pitch_angle[ptch_idx_0:ptch_idx_max+1]).round(decimals=3)
        factor = np.sin(pitches)*np.cos(pitches)
        J_E_pos = np.pi * simpson(y=sweepData[:, ptch_idx_0:ptch_idx_max+1] * factor.round(decimals=3), x=pitches)

        # Total the flux per energy
        J_E_total = -1*J_E_neg+J_E_pos # put a negative sign to counteract the "reverse" integration

        # determine the energy flux
        energy_flux_earthward[tmeIdx] = simpson(J_E_total, Energy)

        # determine the characteristic energy
        characteristic_energy[tmeIdx] = simpson(Energy[engy_idx_GR2001:] * J_E_total[engy_idx_GR2001:], Energy[engy_idx_GR2001:]) / simpson(J_E_total[engy_idx_GR2001:], Energy[engy_idx_GR2001:])

        # determine the energy flux for GR2001
        energy_flux_earthward_GR2001[tmeIdx] = simpson(J_E_total[engy_idx_GR2001:], Energy[engy_idx_GR2001:])

    # form the output dictionary
    data_dict_output = {
        'Epoch':deepcopy(data_dict_ACI['Epoch']),
        'ts2_l2_ace_energy':deepcopy(data_dict_ACI['ts2_l2_aci_energy']),
        'characteristic_energy_GR2001': [np.array(characteristic_energy), {'DEPEND_0':'Epoch', 'UNITS':'eV'}],
        'earthward_energy_flux_eV': [np.array(energy_flux_earthward), {'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}],
        'earthward_energy_flux_ergs': [np.array(energy_flux_earthward)/stl.erg_to_eV, {'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}],
        'earthward_energy_flux_GR2001_eV': [np.array(energy_flux_earthward_GR2001), {'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}],
        'earthward_energy_flux_GR2001_ergs': [np.array(energy_flux_earthward_GR2001)/stl.erg_to_eV,{'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}]
    }

    # --- OUTPUT ---
    output_path = f'{DataPaths.TRACERS_data_folder}/ACI/'+f'{year}/{month}/{day}/ts{SC}/ts2_l2_aci_energy_flux_{year}{month}{day}.cdf'
    stl.outputDataDict(output_path, data_dict=data_dict_output)

# -- EXECUTE ---
ACI_L2_to_energy_flux()


