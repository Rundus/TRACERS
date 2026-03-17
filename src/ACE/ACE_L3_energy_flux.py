# --- ACES_L3_energy_flux ---
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
Emin_R87 = 80 # For Robinson 1987, he suggests only using electron energies above 500 eV since below this tends to ionize the F-Region

# Calculate the rough pitch angle to integrate up to
lat = [78.4]*2
long = [-144.7]*2
alt = [600, 200]
times = [dt.datetime(2024,9,27,22,18)]*2
B_ENU = stl.CHAOS(lat,long,alt,times)
B = [np.linalg.norm(vec) for vec in B_ENU]
pitch_max_val = np.degrees(np.arcsin(np.sqrt(B[0]/B[1])))

def ACE_L3_to_energy_flux():

    # Which data
    year='2025'
    month = '09'
    day = '27'
    SC = '2'

    # load the ACE data
    data_files = glob(f'{DataPaths.TRACERS_data_folder}/ACE/' +f'{year}/{month}/{day}/ts{SC}/*pitch-angle-dist_*')
    data_dict_ACE = stl.loadDictFromFile(data_files[0])


    # --- --- --- --- --- --- --- --- ---
    # --- Calculate the Energy Flux ---
    # --- --- --- --- --- --- --- --- ---

    # Get the detector info
    epoch = data_dict_ACE['Epoch'][0]
    Energy = data_dict_ACE['ts2_l3_ace_energy'][0]
    pitch_angle = data_dict_ACE['ts2_l3_ace_pitch_angle'][0]
    diffEFlux = data_dict_ACE['ts2_l3_ace_pitch_def'][0]

    # Determine the index of energy values to integrate over for the robisnon formulae
    engy_idx_R87 = np.abs(Energy - Emin_R87).argmin()
    Energy_R87 = Energy[:engy_idx_R87+1]

    # Determine the index of pitch angle values to integrate over for the Robinson Formulae
    ptch_idx_max = np.abs(pitch_angle - pitch_max_val).argmin()

    # loop through each time and calculate the energy flux
    earthward_energy_flux = np.zeros(shape=(len(epoch)))
    earthward_energy_flux_R87 = np.zeros(shape=(len(epoch)))
    characteristic_energy = np.zeros(shape=(len(epoch)))
    characteristic_energy_R87 = np.zeros(shape=(len(epoch)))

    # for tmeIdx in tqdm(range(len(epoch))):
    for tmeIdx in tqdm(range(len(epoch))):

        sweepData = diffEFlux[tmeIdx]
        sweepData[sweepData<0] = 0 #ensure no flux is negative

        # "Integrate" over solid angle for each energy sweep
        factor = np.sin(np.radians(pitch_angle))*np.cos(np.radians(pitch_angle))
        prepared_data = sweepData*factor.round(decimals=3) # remove any negative signs which only occur due to precision issues
        J_E = 2*np.pi*simpson(y=prepared_data[:,:ptch_idx_max+1], x=np.radians(pitch_angle)[:ptch_idx_max+1], axis=1) #integrate over sterdians, but only the pitch angles between 0 to 90-ish

        # determine the total energy flux
        earthward_energy_flux[tmeIdx] = -1*simpson(J_E,Energy)

        # determine the characteristic energy
        characteristic_energy[tmeIdx] = simpson(Energy*J_E,Energy)/simpson(J_E,Energy)

        # determine the robinson parallel energy flux for Robinson 1987
        J_E_robinson = 2*np.pi*simpson(y=prepared_data[:,:ptch_idx_max+1],  x=np.radians(pitch_angle[:ptch_idx_max+1]),axis=1)
        earthward_energy_flux_R87[tmeIdx] = -1*simpson(J_E_robinson[:engy_idx_R87+1], Energy_R87) # put a -1 because the energy values decend instead of acend
        characteristic_energy_R87[tmeIdx] = simpson(J_E_robinson[:engy_idx_R87+1]*Energy_R87, Energy_R87)/simpson(J_E_robinson[:engy_idx_R87+1], Energy_R87)


    # form the output dictionary
    data_dict_output = {
        'Epoch':deepcopy(data_dict_ACE['Epoch']),
        'ts2_l3_ace_energy':deepcopy(data_dict_ACE['ts2_l3_ace_energy']),
        'characteristic_energy': [np.array(characteristic_energy), {'DEPEND_0':'Epoch', 'UNITS':'eV'}],
        'characteristic_energy_R87': [np.array(characteristic_energy_R87), {'DEPEND_0': 'Epoch', 'UNITS': 'eV'}],
        'earthward_energy_flux_R87' : [np.array(earthward_energy_flux_R87),{'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}], # put a negative sign so that postive flux means precipitating
        'earthward_energy_flux_R87_ergs': [np.array(earthward_energy_flux_R87)/stl.erg_to_eV,{'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}],
        'earthward_energy_flux_eV': [np.array(earthward_energy_flux), {'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}],
        'earthward_energy_flux_ergs': [np.array(earthward_energy_flux)/stl.erg_to_eV, {'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}]
    }

    # --- OUTPUT ---
    output_path = f'{DataPaths.TRACERS_data_folder}/ACE/' + f'{year}/{month}/{day}/ts{SC}/ts2_l3_ace_energy_flux_{year}{month}{day}.cdf'
    stl.outputDataDict(output_path, data_dict=data_dict_output)



# -- EXECUTE ---
ACE_L3_to_energy_flux()


