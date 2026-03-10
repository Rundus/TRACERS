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

from .ace_file_readers import (read_ace_l2_files, read_ace_l3_files)
from ..general.misc_functions import (determine_datetime_type, getTime)




# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

# ***************************************************************************

def retrieve_ace_data(year,month,day,spacecraft,level,local_dir=None):
    
    """
    This routine retrieves ACE data from the UIowa server. 
    
    Required inputs are year, month, and day of the file you wish to retreive, as well as the spacecraft 
    name ('1' or '2') and data product level (l2 or l3).
    
    This routine will download the most recent file for that date to a local subdirectory, which
    is indicated as the "local_dir" variable. This is defaulted to ./data/TS1(2)/ACE/LL/YYYY/MM.
    """

    level = level.lower()
    if local_dir is None:
        cwd = os.getcwd()
        local_dir = f'{cwd}/data/TS{spacecraft}/ACE/{level}/{year}/{month}/'
        # checking local directory structure. creating if not already in existence
        # ./data/TS1(2)/ACE/LL/YYYY/MM/
        if os.path.exists(f'{cwd}/data/') is False:
            os.mkdir(f'{cwd}/data/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACE/') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACE/')   
        if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/{level}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACE/{level}')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACE/{level}/{year}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACE/{level}/{year}')
        if os.path.exists(f'{cwd}/data/TS{spacecraft}/ACE/{level}/{year}/{month}') is False:
            os.mkdir(f'{cwd}/data/TS{spacecraft}/ACE/{level}/{year}/{month}')

            
    base_url = f'https://tracers-portal.physics.uiowa.edu/teams/flight/ACE/ts{spacecraft}/{level}'
    
    date_url = f'{base_url}/{year}/{month}'
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
        print(f"No ACE {spacecraft.upper()} files for {ymd}!")
    
    return None 




# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
# ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

# outpath/ts{1 or 2}_l2_ace_def_{datestring}_v{version}.cdf
class ACE_L2(getTime):
    """This class will initiate and load ACE L2 data. Inputs are start and stop time in string format, as well as
    the spacecraft you wish to analyze ('1' or '2'). After initializing the class, load the data by employing read_data().

    
    Keys: 'energy', 'def', 'counts', 'bg_counts', 'UTC', 'anode', 'DT', 'start_time', 'end_time', 'spacecraft'
    T x E (49) x 21 (A)

    ex: 
    ace = ACE_L2('2025-11-19/12:00','2025-11-19/13:00','2')
    ace_l2 = ace.read_data()
    
    """
    # *******************************************************************
    def __init__(self,t0,tf,spacecraft):
        super().__init__(t0=t0,tf=tf,spacecraft=spacecraft)  
        self.spacecraft = spacecraft
    # *******************************************************************
    def read_data(self,local_dir=None):
        # Creating subdirectories for data that match format of server, finding local files,
        # and downloading ACE files where need be.

        files2load = []
        for d in range(len(self.date_list)):
            ds = self.date_list[d]
            year = ds.split('/')[0]
            month = ds.split('/')[1]
            day = ds.split('/')[2]
            date_string = year+month+day

            cwd = os.getcwd()
    
            if local_dir is None:
                local_dir = f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}/{month}/'
                # checking local directory structure. creating if not already in existence
                # ./data/TS1(2)/ACE/LL/YYYY/MM/
                if os.path.exists(f'{cwd}/data/') is False:
                    os.mkdir(f'{cwd}/data/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/')   
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l2') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l2')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}/{month}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}/{month}')

                date_dir = f'{cwd}/data/TS{self.spacecraft}/ACE/l2/{year}/{month}'

            else:
                date_dir = local_dir
           
            
            
            f2find = f'{date_dir}/ts{self.spacecraft}_l2_ace_def**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find}').read()
            ace_local_files = out.split('\n')[0:-1]
            
            if len(ace_local_files) > 0:
                files2load.append(ace_local_files[-1])
            else:
                n = retrieve_ace_data(year,month,day,self.spacecraft,'l2',local_dir=local_dir)
                f2find = f'{date_dir}/ts{self.spacecraft}_l2_ace_def**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find}').read()
                ace_local_files = out.split('\n')[0:-1]
                files2load.append(ace_local_files[-1])
                
        self.filenames = files2load
        
        # Loading in data from each L2 CDF
        ace_dict = read_ace_l2_files(files2load, start=self.start, end=self.end)

        return ace_dict

    
    # *******************************************************************



class ACE_L3(getTime):
    # *******************************************************************
    def __init__(self,t0,tf,spacecraft):
        super().__init__(t0=t0,tf=tf,spacecraft=spacecraft)   
    # *******************************************************************
    def read_data(self,local_dir=None):
        # Creating subdirectories for data that match format of server, finding local files,
        # and downloading ACE files where need be.

        files2load = []
        for d in range(len(self.date_list)):
            ds = self.date_list[d]
            year = ds.split('/')[0]
            month = ds.split('/')[1]
            day = ds.split('/')[2]
            date_string = year+month+day

            cwd = os.getcwd()
    
            if local_dir is None:
                local_dir = f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}/{month}/'
                # checking local directory structure. creating if not already in existence
                # ./data/TS1(2)/ACE/LL/YYYY/MM/
                if os.path.exists(f'{cwd}/data/') is False:
                    os.mkdir(f'{cwd}/data/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/')   
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l3') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l3')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}')
                if os.path.exists(f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}/{month}') is False:
                    os.mkdir(f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}/{month}')

                date_dir = f'{cwd}/data/TS{self.spacecraft}/ACE/l3/{year}/{month}'

            else:
                date_dir = local_dir
            
            f2find = f'{date_dir}/ts{self.spacecraft}_l3_ace_pitch**{date_string}**.cdf'
            out = os.popen(f'ls -rt {f2find}').read()
            ace_local_files = out.split('\n')[0:-1]
            
            if len(ace_local_files) > 0:
                files2load.append(ace_local_files[-1])
            else:
                n = retrieve_ace_data(year,month,day,self.spacecraft,'l3',local_dir=local_dir)
                f2find = f'{date_dir}/ts{self.spacecraft}_l3_ace_pitch**{date_string}**.cdf'
                out = os.popen(f'ls -rt {f2find}').read()
                ace_local_files = out.split('\n')[0:-1]
                files2load.append(ace_local_files[-1])
                
        self.filenames = files2load
        
        # Loading in data from each L3 CDF
        ace_dict = read_ace_l3_files(files2load, start=self.start, end=self.end)


        return ace_dict

# ***************************************************************************




    
