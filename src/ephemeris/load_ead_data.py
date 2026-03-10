from src.data_routines.lib.general.ead_file_loading import *

target_directory = '/home/connor/Data/TRACERS/ead/2025/09/27/ts2'
UTC_start = '2025-09-27/22:12:00'
UTC_end = '2025-09-27/22:22:00'

ead = EADload(UTC_start,UTC_end,'2').read_data(params=['lat_geo','lon_geo','lat_geod','lon_geod','alt_geod','mlat','mlt','r_sm'],local_dir=target_directory)
