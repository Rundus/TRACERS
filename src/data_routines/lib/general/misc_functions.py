import datetime as dt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def aurora_cmap():
    """Return a colormap with aurora like colors"""
    stops = {'red': [(0.00, 0.1725, 0.1725),
                     (0.50, 0.1725, 0.1725),
                     (1.00, 0.8353, 0.8353)],

             'green': [(0.00, 0.9294, 0.9294),
                       (0.50, 0.9294, 0.9294),
                       (1.00, 0.8235, 0.8235)],

             'blue': [(0.00, 0.3843, 0.3843),
                      (0.50, 0.3843, 0.3843),
                      (1.00, 0.6549, 0.6549)],

             'alpha': [(0.00, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (1.00, 1.0, 1.0)]}

    return LinearSegmentedColormap('aurora', stops)


def determine_datetime_type(date_string):
    formats = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",\
               "%d/%m/%Y %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y",\
               "%d/%m/%Y", "%Y-%m-%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S", \
               "%Y-%m-%d/%H:%M:%S","%Y%m%d/%H:%M:%S","%Y%m%d-%H:%M:%S",\
               "%Y-%m-%d/%H:%M:%S.%f", '%Y-%m-%d/%H:%M',"%Y%m%d/%H:%M"]
    for fmt in formats:
        try:
            dt.datetime.strptime(date_string, fmt)
            return fmt
        except ValueError:
            continue
    return None

def format_tick_intervals(ax,tick_intervals,interval=None,rotation=None):

    if tick_intervals  == 'second':
        if interval is None:
            interval = 5 
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) 
        ax.xaxis.set_tick_params(rotation=rotation)    
    if tick_intervals == 'minute':
        if interval is None:
            interval = 5 
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        ax.xaxis.set_tick_params(rotation=rotation)

    elif tick_intervals == 'hour':
        if interval is None:
            interval = 5
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d/%H:%M'))   
        ax.xaxis.set_tick_params(rotation=rotation)

        
    elif tick_intervals == 'day':
        if interval is None:
            interval = 1
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
        ax.xaxis.set_tick_params(rotation=rotation)

    elif tick_intervals == 'month':
        if interval is None:
            interval = 1
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))   
        ax.xaxis.set_tick_params(rotation=rotation)

    elif tick_intervals == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))   
        ax.xaxis.set_tick_params(rotation=rotation)

    else:
        print('please input valid time interval!')
        exit()
    return ax




class getTime:
    def __init__(self,t0,tf,spacecraft):
        self.spacecraft = spacecraft # ts1 or ts2 for the satellite number
        dt_fmt = determine_datetime_type(t0)
        self.t0 = t0
        self.tf = tf
        self.start = dt.datetime.strptime(t0,dt_fmt)
        self.end = dt.datetime.strptime(tf,dt_fmt)
        
        sds = f'{self.start.year}-{self.start.month:02}-{self.start.day}'
        eds =  f'{self.end.year}-{self.end.month:02}-{self.end.day}'
        self.date_list = pd.date_range(sds,eds,freq='D').strftime("%Y/%m/%d").tolist()
        self.date_strings = pd.date_range(sds,eds,freq='D').strftime("%Y%m%d").tolist()


def interpolate_mlt(utc, mlt, new_times):
    """
    This function interpolates magnetic local time from EAD files onto a new time series.
    This interpolation takes care of the discontinuity that occurs going from 23 --> 1 h.
    Inputs:
    UTC = seconds since 1/1/1970
    MLT = array of magnetic local times
    new_times = array of UTC times you want new MLT values at
    Returns:
    Array of interpolated MLT values with same shape as new_times
    """
    # Need to do some conversions of MLT to make sure interpolation goes correctly from 23 --> 1
    angles = 2.*np.pi*mlt/24.
    xx = np.cos(angles)
    yy = np.sin(angles)
    fx = interp1d(utc, xx, kind='linear')
    fy = interp1d(utc, yy, kind='linear')
    x_new = fx(new_times)
    y_new = fy(new_times)
    angles_new = np.arctan2(y_new, x_new)
    interp_mlt = (angles_new*24./(2.*np.pi)) % 24.
    return interp_mlt