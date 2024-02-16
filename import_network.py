import numpy as np
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

def network_import():
    
    net = pp.create_empty_network()
    
    buses = pd.read_excel("network.xlsx", sheet_name="bus")
    cable = pd.read_excel("network.xlsx", sheet_name="cable")
    cable = cable.dropna() #drops empty rows
    trafo = pd.read_excel("network.xlsx", sheet_name="trafo")
    sgen = pd.read_excel("network.xlsx", sheet_name="sgen")
    loads = pd.read_excel("network.xlsx", sheet_name="load")
    
    net = pp.create_empty_network()
    #plot.simple_plot(net)
    
    #missing standard types: 
    line_data = {'c_nf_per_km': 539.9999895153535, 'r_ohm_per_km': 0.078, 'x_ohm_per_km': 0.0942,'max_i_ka': 0.535, 'type': 'cs', 'q_mm2': None, 'alpha': None}
    pp.create_std_type(net, line_data, "NA2XS2Y 1x400 RM/25 6/10 kV", element='line')
    line_data = {'c_nf_per_km': 479.99857596970895, 'r_ohm_per_km': 0.1, 'x_ohm_per_km': 0.0974, 'max_i_ka': 0.47100000000000003, 'type': 'cs', 'q_mm2': None, 'alpha': None}
    pp.create_std_type(net, line_data, "NA2XS2Y 1x300 RM/25 6/10 kV", element='line')
    trafo_data = {'i0_percent': 0.04, 'pfe_kw': 22.0, 'vkr_percent': 0.32, 'sn_mva': 63.0, 'vn_lv_kv': 10.0, 'vn_hv_kv': 110.0, 'vk_percent': 18.0, 'shift_degree': 150.0, 'vector_group': None, 'tap_side': 'hv', 'tap_neutral': 0, 'tap_min': -9, 'tap_max': 9, 'tap_step_degree': 0.0, 'tap_step_percent': 1.5, 'tap_phase_shifter': False}
    pp.create_std_type(net, trafo_data, "63 MVA 110/10 kV YNd5", element='trafo')
    
    for idx in range(len(buses.name)):
        geo = (buses.x[idx], buses.y[idx])
        pp.create_bus(net, buses.vn_kv[idx], name=buses.name[idx], index=buses.idx[idx], geodata=geo, in_service=buses.in_service[idx], max_vm_pu=buses.max_vm_pu[idx], min_vm_pu=buses.min_vm_pu[idx])
        
    for idx in range(len(buses.name)): #since we have some cables that are disconnected we dont use them therefore i use the amount of buses = 136
        pp.create_line(net, cable.from_bus[idx], cable.to_bus[idx], cable.length_km[idx], cable.std_type[idx], name=cable.name[idx])
    
    for idx in range(len(trafo.name)):
        pp.create_transformer(net, trafo.hv_bus[idx], trafo.lv_bus[idx], trafo.std_type[idx], max_loading_percent=trafo.max_loading_percent[idx])#, name=trafo.name[idx], tap_pos=trafo.tap_pos[idx], index=idx, parallel=trafo.parallel[idx], df=trafo.df[idx])
        
    for idx in range(len(sgen.name)):
        pp.create_sgen(net, sgen.bus[idx], sgen.p_mw[idx], q_mvar=sgen.q_mvar[idx], sn_mva=sgen.sn_mva[idx], name=sgen.name[idx], index=idx)#scaling=sgen.scaling[idx], max_p_mw=sgen.max_p_mw[idx], min_p_mw=sgen.min_p_mw[idx], max_q_mvar=sgen.max_q_mvar[idx], min_q_mvar=sgen.min_q_mvar[idx])
    
    for idx in range(len(loads.name)):
        pp.create_load(net,loads.bus[idx], loads.p_mw[idx], q_mvar=loads.q_mvar[idx], name=loads.name[idx])#, const_z_percent=loads.const_z_percent[idx], const_i_percent=loads.const_i_percent[idx], sn_mva=loads.sn_mva[idx], #scaling=loads.scaling[idx], index=idx, #in_service=loads.in_service[idx], #max_p_mw=loads.max_p_mw[idx], min_p_mw=loads.min_p_mw[idx], max_q_mvar=loads.max_q_mvar[idx], min_q_mvar=loads.min_q_mvar[idx])
    
    load_prfs = loads.profile
    renew_prfs = sgen.profile
    
    return net, load_prfs, renew_prfs

if __name__ == "__main__":
    net, l, r = network_import()