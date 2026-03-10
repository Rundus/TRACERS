from bs4 import BeautifulSoup
import cdflib
import copy 
import datetime as dt
from glob import iglob
import numpy as np
import os 
import pandas as pd 
import requests
from scipy.interpolate import interp1d
import sys
from tqdm import tqdm

from .efi_file_readers import (read_efi_l2_files)
from ..general.misc_functions import (determine_datetime_type, getTime)

def retrieve_efi_data(year,month,day,spacecraft,level,local_dir=None):
    """
    This routine retrieves EFI data from the UIowa server. 
    
    Required inputs are year, month, and day of the file you wish to retreive, as well as the spacecraft 
    name ('1' or '2') and data product level (l2).
    
    This routine will download the most recent file for that date to a local subdirectory, which
    is indicated as the "local_dir" variable. This is defaulted to ./data/TS1(2)/EFI/LL/YYYY/MM/DD.
    """

    level = level.lower()
    if local_dir is None:
        cwd = os.getcwd()
        local_dir = f'{cwd}/data/TS{spacecraft}/EFI/{level}/{year}/{month}/'
        # checking local directory structure. creating if not already in existence
        # ./data/TS1(2)/EFI/LL/YYYY/MM/DD
        if os.path.exists(f'{cwd}/data/') is False:
            os.mkdir(f'{cwd}/data/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/EFI/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/EFI/')   
        if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/{level}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/EFI/{level}')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/EFI/{level}/{year}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/EFI/{level}/{year}')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/EFI/{level}/{year}/{month}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/EFI/{level}/{year}/{month}')

    base_url = f'https://tracers-portal.physics.uiowa.edu/teams/flight/EFI/{level}/ts{spacecraft}'
    
    date_url = f'{base_url}/{year}/{month}/{day}'
    print(date_url)

    page = requests.get(date_url,auth=('tracers-sot','SciOpsTeamFlight!'))
    data = page.text
    soup = BeautifulSoup(data,"html.parser")
    ds = f'{year}{month}{day}'
    all_strings = soup.find_all('a')
    idx_eac = []
    idx_ehf = []
    idx_hsk = []
    idx_vdc = []
    for i in range(len(all_strings)):
        string_name = all_strings[i].get('href')
        if ds in string_name:
            if 'eac' in string_name:
                idx_eac.append(i)
            elif 'ehf' in string_name:
                idx_ehf.append(i)
            elif 'hsk' in string_name:
                idx_hsk.append(i)
            elif 'vdc' in string_name:
                idx_vdc.append(i)

    # Retrieve EAC - presumes if there's EDC, there is EAC/HSK/EHF data
    if len(idx_eac) > 0:
        day_file = all_strings[idx_eac[-1]].get('href')
        sys.stdout.write('\nDownloading '+f'{day_file}'+'\n')
        file_url_path = date_url + '/' + day_file
        local_file_path = local_dir + '/' + day_file            
        r = requests.get(file_url_path,auth=('tracers-sot','SciOpsTeamFlight!'))
        with open(local_file_path,'wb') as df:
            df.write(r.content)               
    else:
        ymd = f'{year}-{month}-{day}'
        print(f"No EFI {spacecraft.upper()} files for {ymd}!")

    # Retrieve VDC
    vdc_file = all_strings[idx_vdc[-1]].get('href')
    sys.stdout.write('\nDownloading '+f'{vdc_file}'+'\n')
    file_url_path = date_url + '/' + vdc_file
    local_file_path = local_dir + '/' + vdc_file            
    r = requests.get(file_url_path,auth=('tracers-sot','SciOpsTeamFlight!'))
    with open(local_file_path,'wb') as df:
        df.write(r.content)

    # Retrieve HSK
    hsk_file = all_strings[idx_hsk[-1]].get('href')
    sys.stdout.write('\nDownloading '+f'{hsk_file}'+'\n')
    file_url_path = date_url + '/' + hsk_file
    local_file_path = local_dir + '/' + hsk_file            
    r = requests.get(file_url_path,auth=('tracers-sot','SciOpsTeamFlight!'))
    with open(local_file_path,'wb') as df:
        df.write(r.content)

    # Retrieve EHF
    ehf_file = all_strings[idx_ehf[-1]].get('href')
    sys.stdout.write('\nDownloading '+f'{ehf_file}'+'\n')
    file_url_path = date_url + '/' + ehf_file
    local_file_path = local_dir + '/' + ehf_file            
    r = requests.get(file_url_path,auth=('tracers-sot','SciOpsTeamFlight!'))
    with open(local_file_path,'wb') as df:
        df.write(r.content)
    
    return None 

class EFI_L2Public(getTime):
    """This class will initiate and load EFI L2 data. Inputs are start and stop time in string format, as well as
    the spacecraft you wish to analyze ('1' or '2'). After initializing the class, load the data by employing read_data().

    ex: 
    efi = EFI_L2Public('2025-11-19/12:00','2025-11-19/13:00','2')
    efi_l2 = efi.read_data(data_prod='eac+vdc')
    
    """
    # *******************************************************************
    def __init__(self,t0,tf,spacecraft):
        super().__init__(t0=t0,tf=tf,spacecraft=spacecraft)  
        self.spacecraft = spacecraft
    # *******************************************************************
    def read_data(self,local_dir=None,data_prod=None):
        # Creating subdirectories for data that match format of server, finding local files,
        # and downloading EFI files where need be.
        # data_prod: which EFI data products to load?
        #   Options: 'vdc', 'eac', 'ehf', or 'hsk' (DC E-field, AC E-field, HF E-field, or Housekeeping)
        #   Note: You can concatenate them, so 'vdc+eac' is a valid option

        files2load = []
        for d in range(len(self.date_list)):
            ds = self.date_list[d]
            year = ds.split('/')[0]
            month = ds.split('/')[1]
            day = ds.split('/')[2]
            date_string = year+month+day

            cwd = os.getcwd()
    
            if local_dir is None:
                local_dir = f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}/{day}/'
                # checking local directory structure. creating if not already in existence
                # ./data/TS1(2)/EFI/LL/YYYY/MM/DD/
                if os.path.exists(f'{cwd}/data/') is False:
                    os.mkdir(f'{cwd}/data/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/EFI/')   
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/l2') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/EFI/l2')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}/{day}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}/{day}')

                date_dir = f'{cwd}/data/TS{self.spacecraft}/EFI/l2/{year}/{month}/{day}'
                print(date_dir)

            else:
                date_dir = local_dir
           
            
            
            f2find_eac = f'{date_dir}/ts{self.spacecraft}_l2_efi_eac**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find_eac}').read()
            eac_local_files = out.split('\n')[0:-1]

            f2find_vdc = f'{date_dir}/ts{self.spacecraft}_l2_efi_vdc**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find_vdc}').read()
            vdc_local_files = out.split('\n')[0:-1]

            f2find_hsk = f'{date_dir}/ts{self.spacecraft}_l2_efi_hsk**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find_hsk}').read()
            hsk_local_files = out.split('\n')[0:-1]

            f2find_ehf = f'{date_dir}/ts{self.spacecraft}_l2_efi_ehf**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find_ehf}').read()
            ehf_local_files = out.split('\n')[0:-1]
            
            if len(eac_local_files) > 0:
                if 'eac' in data_prod:
                    files2load.append(eac_local_files[-1])
                if 'vdc' in data_prod:
                    files2load.append(vdc_local_files[-1])
                if 'ehf' in data_prod:
                    files2load.append(ehf_local_files[-1])
                if 'hsk' in data_prod:
                    files2load.append(hsk_local_files[-1])
            else:
                n = retrieve_efi_data(year,month,day,self.spacecraft,'l2_public',local_dir=local_dir)

                f2find = f'{date_dir}/ts{self.spacecraft}_l2_efi_eac**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find}').read()
                eac_local_files = out.split('\n')[0:-1]
                if 'eac' in data_prod:
                    files2load.append(eac_local_files[-1])

                f2find_vdc = f'{date_dir}/ts{self.spacecraft}_l2_efi_vdc**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find_vdc}').read()
                vdc_local_files = out.split('\n')[0:-1]
                if 'vdc' in data_prod:
                    files2load.append(vdc_local_files[-1])
                
                f2find_hsk = f'{date_dir}/ts{self.spacecraft}_l2_efi_hsk**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find_hsk}').read()
                hsk_local_files = out.split('\n')[0:-1]
                if 'hsk' in data_prod:
                    files2load.append(hsk_local_files[-1])
                
                f2find_ehf = f'{date_dir}/ts{self.spacecraft}_l2_efi_ehf**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find_ehf}').read()
                ehf_local_files = out.split('\n')[0:-1]
                if 'ehf' in data_prod:
                    files2load.append(ehf_local_files[-1])
                
        self.filenames = files2load
        print(files2load)
        
        # Loading in data from each L2 CDF
        efi_dict = read_efi_l2_files(files2load, start=self.start, end=self.end, data_prod=data_prod)

        return efi_dict

    
    # *******************************************************************
