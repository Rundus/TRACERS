import cdflib
import datetime as dt
import numpy as np
import pandas as pd

# ***********************************************************************************

def read_ace_l2_files(files2load,start=None,end=None):
    """
    Reads in a specific user indicated ACE L2 file. Outputs a dictionary with data
    Keys: utc, dt, def, energy, anode, counts, bg_counts
    T x E (49) x 21 (A)
    """
    # Loading in data from each L2 CDF
    data_dict = {'utc':[],'dt':[],'def':[],'energy':[],\
                'anode':[],'counts':[], 'bg_counts':[]}
    spacecraft = (files2load[0].split('/')[-1]).split('_')[0]
                           
    for file in files2load:
        cdf_file = cdflib.CDF(file)
        print(file)            
        # Converting from nonsensical time to UTC.
        # These are the times at the midpoint of each energy sweep.
        dt_from_cdf = cdflib.cdfepoch.to_datetime(cdf_file['Epoch'])
        dts = pd.to_datetime(dt_from_cdf)
        utcs = [(x - pd.Timestamp('1970-01-01'))/pd.Timedelta('1s') for x in dts]
        
        energy_var = f'{spacecraft}_l2_ace_energy'
        energies = cdf_file[energy_var] # [20 - 14,290] eV; 49 centroid energies

        # Angle at the center of each ACE anode in the spacecraft frame. 
        # (Tracers Satellite Coordinate System)
        # Spans -10 to 190
        anode_angle_var = f'{spacecraft}_l2_ace_TSCS_anode_angle'
        anode_angle = cdf_file[anode_angle_var] # anode angles from the +Z axis in TSCS

        
        # Differential energy flux (T x E x A)
        def_var = f'{spacecraft}_l2_ace_def'
        electron_def = cdf_file[def_var]

        # Counts (T x E x A)
        count_var = f'{spacecraft}_l2_ace_counts'
        electron_counts = cdf_file[count_var]

        bg_count_var = f'{spacecraft}_l2_ace_background_counts'
        electron_bg_counts = cdf_file[count_var]
        
        data_dict['def'].append(electron_def)
        data_dict['dt'].append(dts)
        data_dict['counts'].append(electron_counts)
        data_dict['bg_counts'].append(electron_bg_counts)
        data_dict['utc'].append(utcs)

    data_dict['anode']=anode_angle
    data_dict['energy'] = energies
    
    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # RESHAPING ARRAYS BASED ON MULTIPLE FILES   
    n_files = len(files2load)
    t_tot = 0
    t_shapes = []
    for i in range(n_files):
        d2 = np.shape(data_dict['dt'][i])[0]
        t_tot = t_tot + int(d2)
        t_shapes.append(d2)

    times = np.zeros(t_tot)
    defs = np.zeros((t_tot,49,21))
    counts = np.zeros((t_tot,49,21))
    bg_counts = np.zeros((t_tot,49,21))
    
    for k in range(n_files):
        if k == 0:
            indices = [0,t_shapes[k]]
        elif k != 0 and k < n_files - 1:
            prior_indices = np.sum(t_shapes[0:k])
            indices = [prior_indices,prior_indices+t_shapes[k]]
        else:
            prior_indices = np.sum(t_shapes[0:k])
            indices = [prior_indices,t_tot]

        times[indices[0]:indices[1]] = data_dict['utc'][k]
        defs[indices[0]:indices[1],:,:] = data_dict['def'][k]
        counts[indices[0]:indices[1],:,:] = data_dict['counts'][k]
        bg_counts[indices[0]:indices[1],:,:] = data_dict['bg_counts'][k]

    nan_idx = list(np.where(np.isfinite(times) == False)[0])

    if len(nan_idx) > 0:
        for idx in nan_idx:
            if idx == 0 :
                ii = [idx+1, idx+2]
                time_res = abs(times[ii[1]] - times[ii[0]])
                new_time = times[ii[0]] - time_res
            elif idx != len(data_dict['utc']) - 1:
                ii = [idx-1,idx+2]
                new_time = (times[ii[0]] + times[ii[1]])/2.
            else:
                ii = [idx-2, idx-1]
                time_res = abs(times[ii[1]] - times[ii[0]])
                new_time = times[ii[1]] + time_res                
            times[idx] = new_time 

    if start is None and end is None:
        f_time = (files2load[0].split('v')[0]).split('_')[4]
        start = dt.datetime.strptime(f_time,'%Y%m%d')
        end = start + dt.timedelta(days=1)
    
    if isinstance(start,str):
        dt_fmt = determine_datetime_type(start)
        start = dt.datetime.strptime(start, dt_fmt)
        end = dt.datetime.strptime(end, dt_fmt)

    
    start_time = (start - dt.datetime(1970,1,1)).total_seconds()
    end_time = (end - dt.datetime(1970,1,1)).total_seconds()
    time_mask = (times >= start_time)&(times <= end_time) 

    ace_dict = {}
    ace_dict['energy'] = data_dict['energy']
    ace_dict['def'] = defs[time_mask,:,:]
    ace_dict['counts'] = counts[time_mask,:,:]
    ace_dict['bg_counts'] = bg_counts[time_mask,:,:]
    ace_dict['UTC'] = times[time_mask]
    ace_dict['anode'] = data_dict['anode']
    ace_dict['DT'] = []
    for elem in ace_dict['UTC']:
        ace_dict['DT'].append(dt.datetime.fromtimestamp(elem,dt.UTC))
    ace_dict['start_time'] = start
    ace_dict['end_time'] = end
    ace_dict['spacecraft'] = spacecraft    
    return ace_dict


# ***********************************************************************************



def read_ace_l3_files(files2load, start=None,end=None):
    """
    This function returns a dictionary structure of L3 pitch angle data for ACE.
    Dictionary keys are as follows:
    - utc: Greenwich mean time in seconds since Jan 1970 (T)
    - energy: ACE channel energies (49)
    - pitch_angle: list of 18 pitch angle values spanning 5-175. These are the bin center values. Each bin spans 10 degrees (18)
    - def: Differential energy flux for all times, energies, and pitch angles (Tx49x18)
    - def_unc: Uncertainty in differential energy fluxes (Tx49x18)
    - counts: Counts for all times, energies, and pitch angles (Tx49x18)
    - bg_counts: Background count (Tx49x18)
    """
    # Loading in data from each L2 CDF
    data_dict = {'utc':[],'energy':[],'pitch_angle':[],\
                'def':[],'def_unc':[],\
                'counts':[],'bg_counts':[]}
    spacecraft = (files2load[0].split('/')[-1]).split('_')[0]
    for file in files2load:
        print(file)
        cdf_file = cdflib.CDF(file)
        dt_from_cdf = cdflib.cdfepoch.to_datetime(cdf_file['Epoch'])
        dts = [pd.Timestamp(x) for x in dt_from_cdf]
        utcs = [(x - pd.Timestamp('1970-01-01'))/pd.Timedelta('1s') for x in dts]
        
        energy_var = f'{spacecraft}_l3_ace_energy'
        energies = cdf_file[energy_var] # [20 - 14,290] eV; 49 centroid energies
        
        # Pitch angle (nominally 180 values)
        pitch_var = f'{spacecraft}_l3_ace_pitch_angle'
        pitch_angle = cdf_file[pitch_var]
        
        # Differential energy flux (T x E x PA)
        def_var = f'{spacecraft}_l3_ace_pitch_def'
        electron_def = cdf_file[def_var]

        # Differential energy flux uncertainty (T x E x PA)
        def_unc_var = f'{spacecraft}_l3_ace_pitch_def_unc'
        electron_def_unc = cdf_file[def_var]

        # Counts (T x E x PA)
        count_var = f'{spacecraft}_l3_ace_pitch_counts'
        electron_counts = cdf_file[count_var]

        # Background counts (T x E x PA)
        bg_var = f'{spacecraft}_l3_ace_pitch_background_counts'
        electron_bg_counts = cdf_file[bg_var]
        
        data_dict['utc'].append(utcs)        
        data_dict['def'].append(electron_def)
        data_dict['def_unc'].append(electron_def_unc)
        data_dict['counts'].append(electron_counts)
        data_dict['bg_counts'].append(electron_bg_counts)

    data_dict['pitch_angle'] = pitch_angle
    data_dict['energy'] = energies


    # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    # RESHAPING ARRAYS BASED ON MULTIPLE FILES   

    n_files = len(files2load)
    t_tot = 0
    t_shapes = []
    for i in range(n_files):
        d2 = np.shape(data_dict['utc'][i])[0]
        t_tot = t_tot + int(d2)
        t_shapes.append(d2)

    times = np.zeros(t_tot)
    pitch_angles = np.zeros(18)
    defs = np.zeros((t_tot,49,18))
    def_uncs = np.zeros((t_tot,49,18))
    counts = np.zeros((t_tot,49,18))
    bg_counts = np.zeros((t_tot,49,18))
    if start is None and end is None:
        f_time = (files2load[0].split('v')[0]).split('_')[4]
        start = dt.datetime.strptime(f_time,'%Y%m%d')
        end = start + dt.timedelta(days=1)
    
    if isinstance(start,str):
        dt_fmt = determine_datetime_type(start)
        start = dt.datetime.strptime(start, dt_fmt)
        end = dt.datetime.strptime(end, dt_fmt)


    start_time = (start - dt.datetime(1970,1,1)).total_seconds()
    end_time = (end - dt.datetime(1970,1,1)).total_seconds()
    
    for k in range(n_files):
        if k == 0:
            indices = [0,t_shapes[k]]
        elif k != 0 and k < n_files - 1:
            prior_indices = np.sum(t_shapes[0:k])
            indices = [prior_indices,prior_indices+t_shapes[k]]
        else:
            prior_indices = np.sum(t_shapes[0:k])
            indices = [prior_indices,t_tot]

        times[indices[0]:indices[1]] = data_dict['utc'][k]
        defs[indices[0]:indices[1],:,:] = data_dict['def'][k]
        def_uncs[indices[0]:indices[1],:,:] = data_dict['def_unc'][k]
        counts[indices[0]:indices[1],:,:] = data_dict['counts'][k]
        bg_counts[indices[0]:indices[1],:,:] = data_dict['bg_counts'][k]


    nan_idx = list(np.where(np.isfinite(times) == False)[0])
    if len(nan_idx) > 0:
        for idx in nan_idx:
            if idx == 0 :
                ii = [idx+1, idx+2]
                time_res = abs(times[ii[1]] - times[ii[0]])
                new_time = times[ii[0]] - time_res
            elif idx != len(ace_dict['time']) - 1:
                ii = [idx-1,idx+2]
                new_time = (times[ii[0]] + times[ii[1]])/2.
            else:
                ii = [idx-2, idx-1]
                time_res = abs(times[ii[1]] - times[ii[0]])
                new_time = times[ii[1]] + time_res                
            times[idx] = new_time 
    time_mask = (times >= start_time)&(times <= end_time) 

    ace_dict = {}
    ace_dict['energy'] = data_dict['energy']
    ace_dict['def'] = defs[time_mask,:,:]
    ace_dict['def_err'] = def_uncs[time_mask,:,:]
    ace_dict['counts'] = counts[time_mask,:,:]
    ace_dict['bg_counts'] = bg_counts[time_mask,:,:]
    ace_dict['pitch_angle'] = data_dict['pitch_angle']
    ace_dict['UTC'] = times[time_mask]

    ace_dict['DT'] = []
    for elem in ace_dict['UTC']:
        ace_dict['DT'].append(dt.datetime.fromtimestamp(elem,dt.UTC))
    ace_dict['start_time'] = start
    ace_dict['end_time'] = end
    ace_dict['spacecraft'] = spacecraft

    return ace_dict



















    