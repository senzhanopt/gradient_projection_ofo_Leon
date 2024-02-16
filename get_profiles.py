import numpy as np
from numpy import savetxt
import pandas as pd
import matplotlib.pyplot as plt
import os
# import numba
# from numba import jit

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot
import simbench as sb
import seaborn as sns

def get_profiles(load_prfs, renew_prfs):
    #get all sb profiles, we only need some so will extract valuable data and remove the rest:
    profiles = sb.get_all_simbench_profiles(0) 
    load_profiles = profiles["load"]
    renew_profiles = profiles["renewables"]
    
    load_prfs_unique = load_prfs.drop_duplicates(ignore_index=False)
    load_prfs_unique = load_prfs_unique.reset_index(drop=True)
    renew_prfs_unique = renew_prfs.drop_duplicates(ignore_index=False)
    renew_prfs_unique = renew_prfs_unique.reset_index(drop=True)
    
    size =len(load_prfs_unique) #load profiles are given as different string in the profiles page vs the loads file this fixes it:
    for k in range(size):
        load_prfs_unique.loc[size+k] = load_prfs_unique[k] + "_qload"
        load_prfs_unique.loc[k] = load_prfs_unique[k] + "_pload"
    
    load_prfs_data = pd.DataFrame() #the relevant simbench profiles
    renew_prfs_data = pd.DataFrame()
    
    for profile1 in load_profiles:
        for profile2 in load_prfs_unique:
            if profile1 == profile2:
                load_prfs_data[profile1] = load_profiles[profile1]
    
    for profile1 in renew_profiles:
        for profile2 in renew_prfs_unique:
            if profile1 == profile2:
                renew_prfs_data[profile1] = renew_profiles[profile1]
    
    return(load_prfs_data, renew_prfs_data)