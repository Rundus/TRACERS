# --- ACES_L3_energy_flux ---
# Calculates the energy flux from the ACE detector
import spaceToolsLib as stl
from glob import glob
import numpy as np
from scipy.stats import energy_distance
from scipy.integrate import simpson
from tqdm import tqdm
from copy import deepcopy



# --- --- --- ---
# --- TOGGLES ---
# --- --- --- ---
Emin_R87 = 500 # For Robinson 1987, he suggests only using electron energies above 500 eV since below this tends to ionize the F-Region
pitch_max_R87 = 90

def ACE_L3_to_energy_flux():

    # Which data
    year='2025'
    month = '09'
    day = '27'
    SC = '2'

    # load the ACE data
    data_files = glob('/home/connor/Data/TRACERS/ACE/' +f'{year}/{month}/{day}/ts{SC}/*.cdf*')
    data_dict_ACE = stl.loadDictFromFile(data_files[0])


    # --- --- --- --- --- --- --- --- ---
    # --- Calculate the Energy Flux ---
    # --- --- --- --- --- --- --- --- ---

    # Get the detector info
    epoch = data_dict_ACE['Epoch'][0]
    Energy = data_dict_ACE['ts2_l3_ace_energy'][0]
    pitch_angle = data_dict_ACE['ts2_l3_ace_pitch_angle'][0]
    diffEFlux = data_dict_ACE['ts2_l3_ace_pitch_def'][0]

    # determine the DeltaE to use for the varphi integrations - DeltaE = the half distance to the next energy value
    deltaEs = []
    for idx, engy in enumerate(Energy):
        if idx == len(Energy) - 1:
            deltaEs.append(Energy[-2] - Energy[-1])
        elif idx == 0:
            deltaEs.append(Energy[0] - Energy[1])
        else:
            lowerE = (Energy[idx] - Energy[idx + 1]) / 2
            highE = (Energy[idx - 1] - Energy[idx]) / 2
            deltaEs.append(lowerE + highE)
    deltaEs = np.array(deltaEs)

    # Determine the index of energy values to integrate over for the robisnon formulae
    engy_idx_R87 = np.abs(Energy - Emin_R87).argmin()

    # Determine the index of pitch angle values to integrate over for the Robinson Formulae
    ptch_idx_R87 = np.abs(pitch_angle - pitch_max_R87).argmin()

    # determine the DeltaPitch to use - DeltaPitch is just the width of the polar angle (pitch angle) in radians
    deltaPitch = np.array([ np.radians(val + 5) - np.radians(val - 5) for val in pitch_angle])

    # loop through each time and calculate the energy flux
    para_energy_flux_perE = np.zeros(shape=(len(epoch),len(Energy)))
    para_energy_flux_total = np.zeros(shape=(len(epoch)))
    para_energy_flux_R87 = np.zeros(shape=(len(epoch),))
    characteristic_energy = np.zeros(shape=(len(epoch)))
    characteristic_energy_R87 = np.zeros(shape=(len(epoch)))

    for tmeIdx in tqdm(range(len(epoch))):

        sweepData = diffEFlux[tmeIdx]

        # "Integrate" over solid angle for each energy sweep
        prepared_data = np.sin(np.radians(pitch_angle))*np.cos(np.radians(pitch_angle))* sweepData
        J_E = 2*np.pi*simpson(y=prepared_data, x=pitch_angle, axis=1)
        para_energy_flux_perE[tmeIdx] = -1*J_E*deltaEs

        # determine the total energy flux
        para_energy_flux_total[tmeIdx] = -1*simpson(J_E,Energy)

        # determine the characteristic energy
        characteristic_energy[tmeIdx] = simpson(Energy*J_E,Energy)/simpson(J_E,Energy)

        # determine the robinson parallel energy flux for Robinson 1987
        J_E_robinson = 2*np.pi*simpson(y=prepared_data[:,:ptch_idx_R87+1],  x=pitch_angle[:ptch_idx_R87+1],axis=1)
        para_energy_flux_R87[tmeIdx] = -1*simpson(J_E_robinson[:Emin_R87+1], Energy[:Emin_R87+1]) # put a negative sign so that postive flux means precipitating
        characteristic_energy_R87[tmeIdx] = simpson(Energy[:Emin_R87+1]*J_E_robinson,Energy[:Emin_R87+1])/simpson(J_E_robinson,Energy[:Emin_R87+1])

    # convert energy flux to ergs/cm^2-s
    total_energy_flux_ergs = para_energy_flux_total/stl.erg_to_eV

    # form the output dictionary
    data_dict_output = {
        'parallel_energy_flux_perE':[np.array(para_energy_flux_perE), {'DEPEND_0':'Epoch','DEPEND_1':'ts2_l3_ace_energy','UNITS':'eV/cm!A2!N-s'}],
        'Epoch':deepcopy(data_dict_ACE['Epoch']),
        'ts2_l3_ace_energy':deepcopy(data_dict_ACE['ts2_l3_ace_energy']),
        'characteristic_energy': [np.array(characteristic_energy), {'DEPEND_0':'Epoch', 'UNITS':'eV'}],
        'characteristic_energy_R87': [np.array(characteristic_energy_R87), {'DEPEND_0': 'Epoch', 'UNITS': 'eV'}],
        'parallel_energy_flux_R87' : [np.array(para_energy_flux_R87),{'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}],
        'parallel_energy_flux_R87_ergs': [np.array(para_energy_flux_R87)/stl.erg_to_eV,{'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}],
        'parallel_energy_flux_eV': [np.array(para_energy_flux_total), {'DEPEND_0': 'Epoch', 'UNITS': 'eV/cm!A2!N-s'}],
        'parallel_energy_flux_ergs': [np.array(total_energy_flux_ergs), {'DEPEND_0': 'Epoch', 'UNITS': 'ergs/cm!A2!N-s'}]
    }

    # --- OUTPUT ---
    output_path = f'/home/connor/Data/TRACERS/science/DesJardin/energy_flux/ACE_ts2_l3_energy_flux_{year}{month}{day}.cdf'
    stl.outputDataDict(output_path,
                       data_dict=data_dict_output)



# -- EXECUTE ---
ACE_L3_to_energy_flux()


