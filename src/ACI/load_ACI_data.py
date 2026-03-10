from src.data_routines.lib.ACI.aci_data_loading import *

target_directory = '/home/connor/Data/TRACERS/ACI/2025/09/27/ts2'
UTC_start = '2025-09-27/22:12:00'
UTC_end = '2025-09-27/22:22:00'

L3_data = ACI_L2(UTC_start,UTC_end,'2').read_data(local_dir=target_directory)