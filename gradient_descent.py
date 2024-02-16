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
from get_profiles import get_profiles
from projection import euclid_opt as proj
from projection import make_model

import time
import gurobipy as gp
from gurobipy import GRB, quicksum

from tqdm import tqdm

def gradient_descent_static(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, static_cs, genincr, Xi, alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results_static, static_start, static_delay, Q_up, Q_low, noise_std_voltage, noise_std_current, noise_std_trafo):           
    P_prev = pd.Series(float(0), index=np.arange(133))
    Q_prev = pd.Series(float(0), index=np.arange(133))
    V_prev = pd.Series(float(0), index=np.arange(135))
    line_prev = pd.Series(float(0), index=np.arange(136))
    trafo_prev = pd.Series(float(0), index=np.arange(2))
    
    noise_bus = pd.Series(float(1), index=np.arange(135))
    noise_line = pd.Series(float(1), index=np.arange(136))
    noise_trafo = pd.Series(float(1), index=np.arange(2))
    
    m, p, q = make_model(Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, V_prev, line_prev, V_up, V_low, sgen_maxP, S_max, trafo_lim, trafo_prev, Q_up, Q_low)
       
    print('benchmark:')
    
    net.sgen['p_mw'][0:133] = sgen_maxP
    net.sgen['q_mvar'][0:133] = pd.Series(float(0), index=np.arange(133))
        
    pp.runpp(net)
    for s in tqdm(range(static_cs)):
        for b in range(len(V_up)):
            results_static["bus{}_vm_pu".format(b)][s] = net.res_bus.vm_pu.values[b]
        for g in range(len(P_opt)):
            results_static["PV{}_p_mw".format(g)][s] = net.res_sgen.p_mw.values[g]
        for l in range(len(line_prev)):
            results_static["line{}_percent".format(l)][s] = net.res_line.loading_percent[l]
        for t in range(len(trafo_prev)):
            results_static["trafo{}_percent".format(t)][s] = net.res_trafo.loading_percent[t]
            
    # pf_res_plotly(net, map_style='light')#, aspectratio=(1.4,1))
     
    print("gradient descent algorithm, a = {}, delay = {}:".format(alpha, static_delay))
    #maximum capacity is same as previous net generation values
    
    if static_start == 0: #we start with 0 output power by PV
        net.sgen['p_mw'][0:133] = pd.Series(float(0), index=np.arange(133))
        P_prev = pd.Series(float(0), index=np.arange(133))
    else: #we start with max output power by PV
        net.sgen['p_mw'][0:133] = sgen_maxP
        P_prev = sgen_maxP
        
    pp.runpp(net)
    V_prev = net.res_bus.vm_pu.values[1:136]
    line_prev = net.res_line.loading_percent/100
    trafo_prev = net.res_trafo.loading_percent/100
    
    for s in tqdm(range(static_cs)):
        if s == 0:
            delay_list = ["V_prev{}".format(d) for d in range(static_delay+1)] + ["line_prev{}".format(d) for d in range(static_delay+1)] + ["trafo_prev{}".format(d) for d in range(static_delay+1)]
            delay_memory = pd.DataFrame(float(0), index=range(max([len(V_prev), len(line_prev), len(trafo_prev)])), columns=delay_list)
            for d in range(static_delay+1):
                delay_memory['V_prev{}'.format(d)][0:len(net.res_bus.vm_pu.values)-1] = V_prev
                delay_memory['line_prev{}'.format(d)][0:len(net.res_line.loading_percent)] = line_prev
                delay_memory['trafo_prev{}'.format(d)][0:len(net.res_trafo.loading_percent)] = trafo_prev
            
        for g in range(len(sgen.name)): #gradient step 
            if g != 133: #skip hydroplant
                grad_fp[g] = 2*(P_prev[g] - sgen_maxP[g])
                grad_fq[g] = 2*Xi*Q_prev[g]
            
        for g in range(len(sgen.name)):
            if g != 133: #to ensure hydroplant is not controlled
                P_opt[g] = P_prev[g] - alpha*grad_fp[g]
                Q_opt[g] = Q_prev[g] - alpha*grad_fq[g]

        m, net.sgen['p_mw'][0:133], net.sgen['q_mvar'][0:133] = proj(m, p, q, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, delay_memory['V_prev{}'.format(static_delay)][0:135], delay_memory['line_prev{}'.format(static_delay)][0:136], V_up, V_low, sgen_maxP, S_max, trafo_lim, delay_memory['trafo_prev{}'.format(static_delay)][0:2]) #voltage from previous iteration, maximum power from profiles, inverter max s from net.sgen['sn_mva']
                        
        pp.runpp(net)
        for b in range(len(V_up)):
            results_static["bus{}_vm_pu_grad".format(b)][s] = net.res_bus.vm_pu.values[b]
        for g in range(len(P_opt)):
            results_static["PV{}_p_mw_grad".format(g)][s] = net.res_sgen.p_mw.values[g]
            results_static["PV{}_q_mvar_grad".format(g)][s] = net.res_sgen.q_mvar.values[g]
        for l in range(len(line_prev)):
            results_static["line{}_percent_grad".format(l)][s] = net.res_line.loading_percent[l]
        for t in range(len(trafo_prev)):
            results_static["trafo{}_percent_grad".format(t)][s] = net.res_trafo.loading_percent[t]
            
        P_prev = net.sgen['p_mw'][0:133]
        Q_prev = net.sgen['q_mvar'][0:133]

        for d in range(static_delay, 0, -1):
            delay_memory['V_prev{}'.format(d)][0:len(net.res_bus.vm_pu.values)-1] = delay_memory['V_prev{}'.format(d-1)][0:len(net.res_bus.vm_pu.values)-1]
            delay_memory['line_prev{}'.format(d)][0:len(net.res_line.loading_percent)] = delay_memory['line_prev{}'.format(d-1)][0:len(net.res_line.loading_percent)]
            delay_memory['trafo_prev{}'.format(d)][0:len(net.res_trafo.loading_percent)] = delay_memory['trafo_prev{}'.format(d-1)][0:len(net.res_trafo.loading_percent)]
        
        if noise_std_voltage != 0:
            delay_memory['V_prev{}'.format(0)][0:135] = np.random.normal(net.res_bus.vm_pu.values[1:136], noise_std_voltage)
            delay_memory['line_prev{}'.format(0)][0:len(net.res_line.loading_percent)] = np.random.normal(net.res_line.loading_percent/100, noise_std_current)
            delay_memory['trafo_prev{}'.format(0)][0:2] = np.random.normal(net.res_trafo.loading_percent/100, noise_std_trafo)
        else:
            delay_memory['V_prev{}'.format(0)][0:135] = net.res_bus.vm_pu.values[1:136]
            delay_memory['line_prev{}'.format(0)][0:len(net.res_line.loading_percent)] = net.res_line.loading_percent/100
            delay_memory['trafo_prev{}'.format(0)][0:2] = net.res_trafo.loading_percent/100
            
    # pf_res_plotly(net)
        
    return results_static, net

def gradient_descent_dynamic(net, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, steps, offset, cs, genincr, loads, load_prfs, renew_prfs, Xi, alpha, P_opt, Q_opt, S_max, sgen_maxP, grad_fp, grad_fq, V_up, V_low, trafo_lim, sgen, results, Q_up, Q_low, noise_std_voltage, noise_std_current, noise_std_trafo, delay):        
    P_prev = pd.Series(float(0), index=np.arange(133))
    Q_prev = pd.Series(float(0), index=np.arange(133))
    P_temp = pd.Series(float(0), index=np.arange(133))
    Q_temp = pd.Series(float(0), index=np.arange(133))
    V_prev = pd.Series(float(0), index=np.arange(135))
    line_prev = pd.Series(float(0), index=np.arange(136))
    trafo_prev = pd.Series(float(0), index=np.arange(2))
    
    noise_bus = pd.Series(float(1), index=np.arange(135))
    noise_line = pd.Series(float(1), index=np.arange(136))
    noise_trafo = pd.Series(float(1), index=np.arange(2))
    
    m, p, q = make_model(Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, V_prev, line_prev, V_up, V_low, sgen_maxP, S_max, trafo_lim, trafo_prev, Q_up, Q_low)
    
    print('benchmark:')
    step = 0
    for t in tqdm(range(offset, (steps+offset))): 
        for l in range(len(loads.name)):
            for prfs in load_prfs.columns:
                if loads.profile[l] in prfs:
                    if "pload" in prfs:
                        net.load['p_mw'][l] = load_prfs[prfs][t]/10 #10% of load bith active and reactive
                    if "qload" in prfs:
                        net.load['q_mvar'][l] = load_prfs[prfs][t]/10
    
        for g in range(len(sgen.name)):
            for prfs in renew_prfs.columns:
                if sgen.profile[g] == prfs:
                    net.sgen['p_mw'][g] = genincr*renew_prfs[prfs][t]*0.9 #90% of generation times genincr
        
        #power flow without control:          
        pp.runpp(net)
        for b in range(len(V_up)):
            results["bus{}_vm_pu".format(b)][step] = net.res_bus.vm_pu.values[b]
        for g in range(len(P_opt)):
            results["PV{}_p_mw".format(g)][step] = net.res_sgen.p_mw.values[g]
            results["PV{}_q_mvar".format(g)][step] = net.res_sgen.q_mvar.values[g]
        for l in range(len(line_prev)):
            results["line{}_percent".format(l)][step] = net.res_line.loading_percent[l]
        for t in range(len(trafo_prev)):
            results["trafo{}_percent".format(t)][step] = net.res_trafo.loading_percent[t]
        step += 1
    
    net.sgen['p_mw'][0:133] = pd.Series(float(0), index=np.arange(133))
    net.sgen['q_mvar'][0:133] = pd.Series(float(0), index=np.arange(133))
    pp.runpp(net)
    V_prev = net.res_bus.vm_pu.values[1:136]
    line_prev = net.res_line.loading_percent/100
    trafo_prev = net.res_trafo.loading_percent[0:2]/100
    
    print("gradient descent algorithm, a = {}, delay = {}:".format(alpha, delay))
    step = 0
    for t in tqdm(range(offset, (steps+offset))): #power flow with gradient descent:  
        if t == offset:
            delay_list = ["V_prev{}".format(d) for d in range(delay+1)] + ["line_prev{}".format(d) for d in range(delay+1)] + ["trafo_prev{}".format(d) for d in range(delay+1)]
            delay_memory = pd.DataFrame(float(0), index=range(max([len(V_prev), len(line_prev), len(trafo_prev)])), columns=delay_list)
            for d in range(delay+1):
                delay_memory['V_prev{}'.format(d)][0:len(net.res_bus.vm_pu.values)-1] = V_prev
                delay_memory['line_prev{}'.format(d)][0:len(net.res_line.loading_percent)] = line_prev
                delay_memory['trafo_prev{}'.format(d)][0:len(net.res_trafo.loading_percent)] = trafo_prev
                        
        for l in range(len(loads.name)): #updating loads from profiles:
            for prfs in load_prfs.columns:
                if loads.profile[l] in prfs:
                    if "pload" in prfs:
                        net.load['p_mw'][l] = load_prfs[prfs][t]/10
                    if "qload" in prfs:
                        net.load['q_mvar'][l] = load_prfs[prfs][t]/10
                        
        for g in range(len(sgen.name)): #getting maximum        
            for prfs in renew_prfs.columns:
                if sgen.profile[g] == prfs:
                    if g != 133: #skip hydroplant
                        sgen_maxP[g] = genincr*renew_prfs[prfs][t]*0.9              
                
        for s in range(cs):
            for g in range(len(sgen.name)): #gradient step 
                if g != 133: #skip hydroplant
                    grad_fp[g] = 2*(P_prev[g] - sgen_maxP[g])
                    grad_fq[g] = 2*Xi*Q_prev[g]
                
            for g in range(len(sgen.name)):
                if g != 133: #to ensure hydroplant is not controlled
                    P_opt[g] = P_prev[g] - alpha*grad_fp[g]
                    Q_opt[g] = Q_prev[g] - alpha*grad_fq[g]

            m, P_prev, Q_prev = proj(m, p, q, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, delay_memory['V_prev{}'.format(delay)][0:135], delay_memory['line_prev{}'.format(delay)][0:136], V_up, V_low, sgen_maxP, S_max, trafo_lim, delay_memory['trafo_prev{}'.format(delay)][0:2]) #voltage from previous iteration, maximum power from profiles, inverter max s from net.sgen['sn_mva']
            
            net.sgen['p_mw'][0:133] = P_prev
            net.sgen['q_mvar'][0:133] = Q_prev
            pp.runpp(net)
            # V_prev = net.res_bus.vm_pu.values[1:136]
            # line_prev = net.res_line.loading_percent/100
            # trafo_prev = net.res_trafo.loading_percent[0:2]/100
            
            for d in range(delay, 0, -1):
                delay_memory['V_prev{}'.format(d)][0:len(net.res_bus.vm_pu.values)-1] = delay_memory['V_prev{}'.format(d-1)][0:len(net.res_bus.vm_pu.values)-1]
                delay_memory['line_prev{}'.format(d)][0:len(net.res_line.loading_percent)] = delay_memory['line_prev{}'.format(d-1)][0:len(net.res_line.loading_percent)]
                delay_memory['trafo_prev{}'.format(d)][0:len(net.res_trafo.loading_percent)] = delay_memory['trafo_prev{}'.format(d-1)][0:len(net.res_trafo.loading_percent)]
            
            delay_memory['V_prev{}'.format(0)][0:135] = net.res_bus.vm_pu.values[1:136]
            delay_memory['line_prev{}'.format(0)][0:len(net.res_line.loading_percent)] = net.res_line.loading_percent/100
            delay_memory['trafo_prev{}'.format(0)][0:2] = net.res_trafo.loading_percent/100
            
        for b in range(len(V_up)):
            results["bus{}_vm_pu_grad".format(b)][step] = net.res_bus.vm_pu.values[b]
        for g in range(len(P_opt)):
            results["PV{}_p_mw_grad".format(g)][step] = net.res_sgen.p_mw.values[g]
            results["PV{}_q_mvar_grad".format(g)][step] = net.res_sgen.q_mvar.values[g]
        for l in range(len(line_prev)):
            results["line{}_percent_grad".format(l)][step] = net.res_line.loading_percent[l]
        for t in range(len(trafo_prev)):
            results["trafo{}_percent_grad".format(t)][step] = net.res_trafo.loading_percent[t]
        step += 1
        
    return results, net


    