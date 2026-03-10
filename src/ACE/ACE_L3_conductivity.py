# --- ACES_L3_conductivity ---
# Calculates the conductivity from the ACE energy flux calculation
import spaceToolsLib as stl
from glob import glob
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def ACE_L3_conductivity():

    # Which data
    year='2025'
    month = '09'
    day = '27'
    SC = '2'

    # load the ACE data
    data_files = glob('/home/connor/Data/TRACERS/science/DesJardin/energy_flux/' +f'*{year}{month}{day}*')
    data_dict_ACE = stl.loadDictFromFile(data_files[0])


    # --- --- --- --- --- --- --- --- --- --- ---- ---
    # --- Calculate the Robinson conductivity values ---
    # --- --- --- --- --- --- --- --- --- --- ---- ---

    # collect some variables
    characteristic_energy_keV = data_dict_ACE['characteristic_energy'][0]/1000 # convert to keV
    parallel_energy_flux = data_dict_ACE['parallel_energy_flux_ergs'][0]

    # Use the formulae
    Sigma_P_R87 = (40*(characteristic_energy_keV) / (16 + np.square(characteristic_energy_keV))) * np.sqrt(parallel_energy_flux)
    Sigma_H_R87 = 0.45*(characteristic_energy_keV)**(0.85)



    # form the output dictionary
    data_dict_output = {
        'Sigma_P_R87':[np.array(Sigma_P_R87), {'DEPEND_0':'Epoch','UNITS':'mho'}],
        'Sigma_H_R87': [np.array(Sigma_H_R87),{'DEPEND_0':'Epoch','UNITS':'mho'}],
        'Epoch':deepcopy(data_dict_ACE['Epoch']),
        'characteristic_energy': deepcopy(data_dict_ACE['characteristic_energy'])
    }

    # --- OUTPUT ---
    output_path = f'/home/connor/Data/TRACERS/science/DesJardin/conductivity/ACE_ts2_l3_conductivity_{year}{month}{day}.cdf'
    stl.outputDataDict(output_path, data_dict=data_dict_output)



# -- EXECUTE ---
ACE_L3_conductivity()


