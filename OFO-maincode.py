import numpy as np
from numpy import loadtxt
from numpy import savetxt
import pandas as pd
import matplotlib.pyplot as plt
import os
import numba

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot
from pandapower.plotting.plotly import pf_res_plotly

import simbench as sb
import seaborn as sns

from import_network import network_import
from sens_matrix import sens_matrix_it
from sens_matrix import sens_matrix_noise
from get_profiles import get_profiles
from projection import euclid_opt as proj
from projection import make_model
from gradient_descent import gradient_descent_static
from gradient_descent import gradient_descent_dynamic

import time
import gurobipy as gp
from gurobipy import GRB, quicksum

from tqdm import tqdm

# %% importing network:

[net, load_prfs, renew_prfs]  = network_import()

#loads & sgen tables valuable for debugging:
loads = pd.read_excel("network.xlsx", sheet_name="load")
sgen = pd.read_excel("network.xlsx", sheet_name="sgen")

#plot.simple_plot(net)
#plot.simple_plot(net, plot_sgens=True)

pp.create_ext_grid(net, 0, vm_pu=1.025) #sets bus 0 as slack bus with voltage 1.025


# %% Data structures
P_opt = pd.Series(float(0), index=np.arange(133))
Q_opt = pd.Series(float(0), index=np.arange(133))
grad_fp = pd.Series()
grad_fq = pd.Series()
sgen_maxP = pd.Series(float(0), index=np.arange(133))
S_max = pd.Series(float(0), index=np.arange(133))

V_up = net.bus['max_vm_pu'][1:136]
V_low = net.bus['min_vm_pu'][1:136]
trafo_lim = net.trafo['max_loading_percent'][:]/100

meas_volt = ["bus{}_vm_pu".format(b) for b in range(135)] + ["bus{}_vm_pu_grad".format(b) for b in range(135)]
meas_sgen = ["PV{}_p_mw".format(g) for g in range(133)] + ["PV{}_p_mw_grad".format(g) for g in range(133)] + ["PV{}_q_mvar".format(g) for g in range(133)] + ["PV{}_q_mvar_grad".format(g) for g in range(133)]
meas_lines = ["line{}_percent".format(l) for l in range(136)] + ["line{}_percent_grad".format(l) for l in range(136)]
meas_trafo = ["trafo{}_percent".format(t) for t in range(2)] + ["trafo{}_percent_grad".format(t) for t in range(2)]
meas_list = meas_volt + meas_sgen + meas_lines + meas_trafo;


# %% control parameters:
Static = 1 # 0 for dynamic, 1 for static

Xi = 0.1

#static & dynamic parameters:
alpha = [0.05] #a large alpha with a large delay can cause instability/undesired behaviour
delay = [0] #delay representing communication delay of measurements/control. 
cs = [100] #for static case this represents total timesteps(only uses first value of list), for dynamic case in-between timesteps
static_start_pos = 0 #0 for starting with 0 power generation by PV, 1 for starting with maximum PV power

#dynamic parameters:
steps = 96 #amount of timesteps 96 is one day
week_offset = 18 #week 18 is in the middle of spring
offset = week_offset*672 #offset for timesteps -> determines which week we choose for generation/load profiles every week is 672 timesteps

#noise:
noise = 0 #0 for no noise added, 1 for running with noise on the measurements of voltage, line currents and transformer currents. 
noise_std_voltage = [0.001, 0.005] #standard deviations of the bus voltage measurements.
noise_std_current = [0.001, 0.005] #standard deviations of the line current measurements.
noise_std_trafo = [0.001, 0.005] #standard deviations of the trafo current measurements.

genincr = 5 #increase factor PV generation

noise_matrices = 0 #1 to add noise to sens. matrices, only works when they are constructed as well!
std_noise_bus =  0
std_noise_lines = 0
std_noise_trafo = 0


for g in range(len(S_max)):
    S_max[g] = sgen['sn_mva'][g]*genincr

#reactive power limits of inverters:
Q_up = GRB.INFINITY
Q_low = -GRB.INFINITY

# %% Import sensitivity matrix Mp = dV/dP and Mq = dV/dQ:

# sens_matrix_it(net) #create the sensitivity matrix using this Staticwhich saves them into csv files.
# sens_matrix_noise(net, std_noise_bus, std_noise_lines, std_noise_trafo) #create the sensitivity matrix with added gaussian noise

if noise_matrices == 0:
    Mp_b = loadtxt('Mp_b_it.csv', delimiter=',')
    Mq_b = loadtxt('Mq_b_it.csv', delimiter=',')
    Mp_l = loadtxt('Mp_l_it.csv', delimiter=',')
    Mq_l = loadtxt('Mq_l_it.csv', delimiter=',')
    Mp_t = loadtxt('Mp_t_it.csv', delimiter=',')
    Mq_t = loadtxt('Mq_t_it.csv', delimiter=',')
else:
    Mp_b = loadtxt('Mp_b_noise_{}.csv'.format(std_noise_bus*1000), delimiter=',')
    Mq_b = loadtxt('Mq_b_noise_{}.csv'.format(std_noise_bus*1000), delimiter=',')
    Mp_l = loadtxt('Mp_l_noise_{}.csv'.format(std_noise_lines*1000), delimiter=',')
    Mq_l = loadtxt('Mq_l_noise_{}.csv'.format(std_noise_lines*1000), delimiter=',')
    Mp_t = loadtxt('Mp_t_noise_{}.csv'.format(std_noise_trafo*1000), delimiter=',')
    Mq_t = loadtxt('Mq_t_noise_{}.csv'.format(std_noise_trafo*1000), delimiter=',')
    
# %% Importing profiles from Simbench:

load_prfs_data, renew_prfs_data = get_profiles(load_prfs, renew_prfs)
    
# %% Static case:

if Static == 1:
    print('Running a static simulation.')
    #change load to 10% of nominal value:
    for l in range(len(net.load)):
        net.load['p_mw'][l] = 0.1*net.load['p_mw'][l]
        net.load['q_mvar'][l] = 0.1*net.load['q_mvar'][l]
    
    sgen_maxP[0:133] = 0.9*net.sgen['p_mw'][0:133]*genincr #90% of max capacity scaled by genincr
    
    for static_delay in delay:
        for static_alpha in alpha:
            if noise:
                for bus_noise in noise_std_voltage:
                    for line_noise in noise_std_current:
                        for trafo_noise in noise_std_trafo:
                            a = int(static_alpha*100)
                            results_static = pd.DataFrame(float(0), index=range(cs[0]), columns=meas_list)
                            results_static, net = gradient_descent_static(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, cs[0], genincr, Xi, static_alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results_static, static_start_pos, static_delay, Q_up, Q_low, bus_noise, line_noise, trafo_noise)
                            
                            results_static.to_csv('results_static_{}-{}-noise-{}-{}-{}.csv'.format(static_delay, a, int(bus_noise*1000), int(line_noise*1000), int(trafo_noise*1000)))
            else:
                a = int(static_alpha*100)
                results_static = pd.DataFrame(float(0), index=range(cs[0]), columns=meas_list)
                
                results_static, net = gradient_descent_static(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, cs[0], genincr, Xi, static_alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results_static, static_start_pos, static_delay, Q_up, Q_low, 0, 0, 0)
            
                results_static.to_csv('results_static_{}-{}.csv'.format(static_delay, a))

# %% Dynamic case:
    
if Static == 0:
    print('Running a dynamic simulation.')
    # compute load/generation values over time needed:
    load_prfs = pd.DataFrame()
    i = 0;
    for prfs in load_prfs_data.columns:
        load_prfs.insert(loc=i, column=prfs, value=load_prfs_data.iloc[offset:(steps+offset)][prfs])
    
    renew_prfs = pd.DataFrame()
    i = 0;
    for prfs in renew_prfs_data.columns:
        renew_prfs.insert(loc=i, column=prfs, value=renew_prfs_data.iloc[offset:(steps+offset)][prfs])
    
    results = pd.DataFrame(float(0), index=range(offset, (steps+offset)), columns=meas_list)

    # Run power flow per iteration:
    for dyn_delay in delay:
        for dyn_alpha in alpha:
            for dyn_cs in cs:  
                if noise:
                    for bus_noise in noise_std_voltage:
                        for line_noise in noise_std_current:
                            for trafo_noise in noise_std_trafo:
                                a = int(dyn_alpha*100)
                                results_dyn = pd.DataFrame(float(0), index=range(steps), columns=meas_list)
                                results_dyn, net = gradient_descent_dynamic(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, steps, offset, dyn_cs, genincr, loads, load_prfs, renew_prfs, Xi, dyn_alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results_dyn, Q_up, Q_low, bus_noise, line_noise, trafo_noise, dyn_delay)
                                
                                results_dyn.to_csv('results_dyn_{}-{}-{}-noise-{}-{}-{}.csv'.format(dyn_delay, a, dyn_cs, int(bus_noise*1000), int(line_noise*1000), int(trafo_noise*1000)))
                else:
                    a = int(dyn_alpha*100)
                    results_dyn = pd.DataFrame(float(0), index=range(steps), columns=meas_list)
                    
                    results_dyn, net = gradient_descent_dynamic(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, steps, offset, dyn_cs, genincr, loads, load_prfs, renew_prfs, Xi, dyn_alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results_dyn, Q_up, Q_low, 0, 0, 0, dyn_delay)
                
                    results_dyn.to_csv('results_dyn_{}-{}-{}.csv'.format(dyn_delay, a, dyn_cs))

    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        