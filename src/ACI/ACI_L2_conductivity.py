# --- ACI_L2_conductivity ---
# Calculates the conductivity from the ACE energy flux calculation
import spaceToolsLib as stl
from glob import glob
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.data_paths import DataPaths
from scipy.interpolate import CubicSpline


def ACI_L2_conductivity():

    # Which data
    year='2025'
    month = '09'
    day = '27'
    SC = '2'

    # load the ACI data
    data_files = glob(f'{DataPaths.TRACERS_data_folder}/ACI/' + f'{year}/{month}/{day}/ts{SC}/ts2_l2_aci_energy_flux_{year}{month}{day}.cdf')
    data_dict_ACI = stl.loadDictFromFile(data_files[0])

    # load the ephemeris data
    data_files_ead = glob(f'{DataPaths.TRACERS_data_folder}/ead/' + f'{year}/{month}/{day}/ts{SC}/*.cdf*')
    data_dict_ead = stl.loadDictFromFile(data_files_ead[0])

    # get the ephemeris data
    lat = data_dict_ead['ts2_ead_lat_geo'][0]
    long = data_dict_ead['ts2_ead_lon_geo'][0]
    alt = data_dict_ead['ts2_ead_altitude_geod'][0]  # in km

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    # --- Determine the geomagnetic field at 110 km along satellite path ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
    # Note: The Galand Formulae need the magnetic field magnitude at 110km. Here we calculate it and
    # interpolate onto the ACI timebase



    # --- --- --- --- --- --- --- --- --- --- ---- ---
    # --- Calculate the Robinson conductivity values ---
    # --- --- --- --- --- --- --- --- --- --- ---- ---

    # collect some variables
    characteristic_energy_keV = data_dict_ACI['characteristic_energy_GR2001'][0]/1000 # convert to keV
    parallel_energy_flux_ergs = data_dict_ACI['earthward_energy_flux_GR2001_ergs'][0]

    # Use the formulae
    B0 = 54E-6 # in tesla
    B = 50000E-9
    B_ratio = B/B0
    Sigma_P_GR2001 = 5.7*np.sqrt(parallel_energy_flux_ergs) * (B_ratio**(-1.45))
    Sigma_H_GR2001 = 2.6*((characteristic_energy_keV)**(0.3)) * np.sqrt(parallel_energy_flux_ergs) * (B_ratio**(-1.90))

    # form the output dictionary
    data_dict_output = {
        'Sigma_P_GR2001':[np.array(Sigma_P_GR2001), {'DEPEND_0':'Epoch','UNITS':'mho'}],
        'Sigma_H_GR2001': [np.array(Sigma_H_GR2001),{'DEPEND_0':'Epoch','UNITS':'mho'}],
        'Epoch':deepcopy(data_dict_ACI['Epoch']),
        'characteristic_energy': deepcopy(data_dict_ACI['characteristic_energy_GR2001'])
    }

    # --- OUTPUT ---
    output_path = f'{DataPaths.TRACERS_data_folder}/ACI/' + f'{year}/{month}/{day}/ts{SC}/ts2_l2_aci_conductivity_{year}{month}{day}.cdf'
    stl.outputDataDict(output_path, data_dict=data_dict_output)



# -- EXECUTE ---
ACI_L2_conductivity()


