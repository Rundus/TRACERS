import cdflib
import datetime as dt
import numpy as np
import pandas as pd

def read_aci_l2_files(files2load, start=None, end=None):
    """
    Reads in a specific user indicated ACI L2 file. Outputs a dictionary with data.
    This function is calledin ACI_L2 class but can also operate autonomously.
    
    Keys: utc, dt, def, energy, anode, counts, bg_counts
    T x E (47) x 16 (A)
    """
    # Loading in data from each L2 CDF
    data_dict = {'utc':[],'dt':[],'def':[],'energy':[],\
                'anode':[],'counts':[]}
    spacecraft = (files2load[0].split('/')[-1]).split('_')[0]
                           
    for file in files2load:
        cdf_file = cdflib.CDF(file)
        print(file)            
        # Converting from nonsensical time to UTC.
        # These are the times at the midpoint of each energy sweep.
        dt_from_cdf = cdflib.cdfepoch.to_datetime(cdf_file['Epoch'])
        dts = pd.to_datetime(dt_from_cdf)
        utcs = [(x - pd.Timestamp('1970-01-01'))/pd.Timedelta('1s') for x in dts]
        
        energy_var = f'{spacecraft}_l2_aci_energy'
        energies = cdf_file[energy_var] # [20 - 14,290] eV; 47 centroid energies

        # Angle at the center of each ACI anode in the spacecraft frame. 
        # (Tracers Satellite Coordinate System)
        anode_angle_var = f'{spacecraft}_l2_aci_tscs_anode_angle'
        anode_angle = cdf_file[anode_angle_var] # anode angles from the +Z axis in TSCS

        
        # Differential energy flux (T x E x A)
        def_var = f'{spacecraft}_l2_aci_tscs_def'
        ion_def = cdf_file[def_var]

        # Counts (T x E x A)
        count_var = f'{spacecraft}_l2_aci_tscs_def_sorted_counts'
        ion_counts = cdf_file[count_var]

        
        data_dict['def'].append(ion_def)
        data_dict['dt'].append(dts)
        data_dict['counts'].append(ion_counts)
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
    defs = np.zeros((t_tot,47,16))
    counts = np.zeros((t_tot,47,16))
    bg_counts = np.zeros((t_tot,47,16))
    
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

    aci_dict = {}
    aci_dict['energy'] = data_dict['energy']
    aci_dict['def'] = defs[time_mask,:,:]
    aci_dict['counts'] = counts[time_mask,:,:]
    aci_dict['UTC'] = times[time_mask]
    aci_dict['anode'] = data_dict['anode']
    aci_dict['DT'] = []
    for elem in aci_dict['UTC']:
        aci_dict['DT'].append(dt.datetime.fromtimestamp(elem,dt.UTC))
    aci_dict['start_time'] = start
    aci_dict['end_time'] = end
    aci_dict['spacecraft'] = spacecraft    
    return aci_dict


