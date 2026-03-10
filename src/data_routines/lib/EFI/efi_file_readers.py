import datetime as dt
from spacepy import pycdf
import bisect


def _read_efi_l2_eac(file, start, end):
    """
    Creates dict from cdf file usable for plotting
    """
    cdf = pycdf.CDF(file)
    start_ind_ts = bisect.bisect_left(cdf['Epoch'], start)
    end_ind_ts = bisect.bisect_left(cdf['Epoch'], end)

    start_ind_spec = bisect.bisect_left(cdf['ts2_l2_eac_packet_start'], start)
    end_ind_spec = bisect.bisect_left(cdf['ts2_l2_eac_packet_start'], end)

    ddict = {}
    ddict['Epoch'] = cdf['Epoch'][start_ind_ts:end_ind_ts]
    ddict['ts2_l2_eac'] = cdf['ts2_l2_eac'][start_ind_ts:end_ind_ts,:]

    ddict['Frequency'] = cdf['Frequency']
    ddict['ts2_l2_eac_packet_start'] = cdf['ts2_l2_eac_packet_start'][start_ind_spec:end_ind_spec]
    ddict['ts2_l2_eac_x_spec'] = cdf['ts2_l2_eac_x_spec'][start_ind_spec:end_ind_spec]
    ddict['ts2_l2_eac_y_spec'] = cdf['ts2_l2_eac_y_spec'][start_ind_spec:end_ind_spec]
    
    return ddict

def _read_efi_l2_vdc(file, start, end):
    """
    Creates dict from cdf file usable for plotting
    """
    ddict = {}
    cdf = pycdf.CDF(file)
    start_ind = bisect.bisect_left(cdf['Epoch'], start)
    end_ind = bisect.bisect_left(cdf['Epoch'], end)

    for k in ['Epoch', 'ts2_l2_vdc_xminus', 'ts2_l2_vdc_xplus', 'ts2_l2_vdc_yminus', 'ts2_l2_vdc_yplus']:
        ddict[k] = cdf[k][start_ind:end_ind]
    
    return ddict
    

def read_efi_l2_files(files2load,start=None,end=None,data_prod=None):
    """
    Reads in a specific user indicated EFI L2 file. Outputs a dictionary with data
    """
    # Loading in data from each L2 CDF
    data_dict = {}
    spacecraft = (files2load[0].split('/')[-1]).split('_')[0]
    data_dict['spacecraft'] = spacecraft

    for dp in data_prod.split('+'):
        # Find the file for that data product (vdc, eac, ehf, and hsk)
        for f in files2load:
            if dp in f:
                relevant_file = f

        if dp=='eac':
            data_dict[dp] = _read_efi_l2_eac(relevant_file,start,end)
        elif dp=='vdc':
            data_dict[dp] = _read_efi_l2_vdc(relevant_file,start,end)
        elif dp=='ehf':
            raise Error('EHF not supported yet :(')
        elif dp=='hsk':
            raise Error('HSK not supported yet :(')
            
    return data_dict
