from bs4 import BeautifulSoup
import cdflib
import datetime as dt
from glob import iglob
import numpy as np
import os 
import pandas as pd 
import requests
import sys

from ..general.misc_functions import (determine_datetime_type, getTime)
from .aci_file_readers import read_aci_l2_files

# ********************************************
# ********************************************
# ********************************************

def retrieve_aci_data(year,month,day,spacecraft,level,local_dir=None):
    
    """
    This routine retrieves ACI data from the UIowa server. Required inputs are 
    year, month, and day of the file you wish to retreive, as well as the spacecraft 
    name ('1' or '2') and data product level ('l2' or 'l3').
    This routine will download the most recent file for that date to a local subdirectory, which
    is indicated as the "local_dir" variable.
    """

    level = level.lower()
    if local_dir is None:
        cwd = os.getcwd()
        local_dir = f'{cwd}/data/TS{spacecraft}/ACI/l2/{year}/{month}/'
        # checking local directory structure. creating if not already in existence
        # ./data/TS1(2)/ACI/LL/YYYY/MM/
        if os.path.exists(f'{cwd}/data/') is False:
            os.mkdir(f'{cwd}/data/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACI/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACI/')   
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACI/l2') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACI/l2')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACI/l2/{year}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACI/l2/{year}')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACI/l2/{year}/{month}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACI/l2/{year}/{month}')



    base_url = f'https://tracers-portal.physics.uiowa.edu/teams/flight/ACI/ts{spacecraft}/{level}/aci/ipd'
    
    date_url = f'{base_url}/' 
    print(date_url)

    
    page = requests.get(date_url,auth=('tracers-sot','SciOpsTeamFlight!'))
    data = page.text
    soup = BeautifulSoup(data,"html.parser")
    ds = f'{year}{month}{day}'
    all_strings = soup.find_all('a')
    idx = []
    for i in range(len(all_strings)):
        string_name = all_strings[i].get('href')
        if ds in string_name:
            idx.append(i)
    if len(idx) > 0:
        day_file = all_strings[idx[-1]].get('href')
        sys.stdout.write('\nDownloading '+f'{day_file}'+'\n')
        file_url_path = date_url + '/' + day_file
        local_file_path = local_dir + '/' + day_file            
        r = requests.get(file_url_path,auth=('tracers-sot','SciOpsTeamFlight!'))
        with open(local_file_path,'wb') as df:
            df.write(r.content)           
        
    else:
        ymd = f'{year}-{month}-{day}'
        print(f"No ACI {spacecraft.upper()} files for {ymd}!")
    
    return None 


# ********************************************
# ********************************************
# ********************************************

class ACI_L2(getTime):
    """
    Class to load and read ACI L2 data.
    Required inputs
    -------------------
    t0 = start time of observation (string in datetime format. i.e., '2025-11-06/12:00:34')
    tf = end time of observation in same format as t0
    spacecraft = '1' or '2' for whichever TRACERS satellite you're aiming to look at.
    """
    def __init__(self,t0,tf,spacecraft):
        super().__init__(t0=t0,tf=tf,spacecraft=spacecraft)
        self.spacecraft = spacecraft

    def read_data(self,local_dir=None):
        """
        Reads data from object generated from class. 
        Optional input
        --------------
        local_dir = user input path where they wish to put data.
            default results in pathways being generated in the form
            ./data/TS{1 or 2}/ACI/l{2 or 3}/{year}/{month}
        """
            
        files2load = []
        for d in range(len(self.date_list)):
            ds = self.date_list[d]
            year = ds.split('/')[0]
            month = ds.split('/')[1]
            day = ds.split('/')[2]
            date_string = year+month+day
            cwd = os.getcwd()
            
            if local_dir is None:
                local_dir = f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}/{month}/'
                # checking local directory structure. creating if not already in existence
                # ./data/TS1(2)/ACI/LL/YYYY/MM/
                if os.path.exists(f'{cwd}/data/') is False:
                    os.mkdir(f'{cwd}/data/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACI/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACI/')   
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACI/l2') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACI/l2')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}/{month}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}/{month}')
                date_dir = f'{cwd}/data/TS{self.spacecraft}/ACI/l2/{year}/{month}'
            else:
                date_dir = local_dir

            
            f2find = f'{date_dir}/ts{self.spacecraft}_l2_aci_**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find}').read()
            aci_local_files = out.split('\n')[0:-1]
            
            if len(aci_local_files) > 0:
                files2load.append(aci_local_files[-1])
            else:
                aci_download = retrieve_aci_data(year,month,day,self.spacecraft,'l2')
                f2find = f'{date_dir}/ts{self.spacecraft}_l2_aci_**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find}').read()
                aci_local_files = out.split('\n')[0:-1]
                files2load.append(aci_local_files[-1])

        self.filenames = files2load
        
        # Loading in data from each L2 CDF
        aci_dict = read_aci_l2_files(files2load, start=self.start, end=self.end)   
        return aci_dict



# ********************************************
# ********************************************
# ********************************************





        
