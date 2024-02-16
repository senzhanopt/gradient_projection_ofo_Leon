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

def sens_matrix_it(net): 
    net.sgen.p_mw *= 5
    net.load.p_mw *= 0.1
    net.load.q_mvar *= 0.1
      
    pp.runpp(net)
    vm_orig = net.res_bus.vm_pu[1:]
    line_orig = net.res_line.loading_percent[:]/100
    trafo_orig = net.res_trafo.loading_percent[:]/100    
    
    Mp_b_it = np.zeros((135, 133))
    Mq_b_it = np.zeros((135, 133))
    Mp_l_it = np.zeros((136, 133))
    Mq_l_it = np.zeros((136, 133))
    Mp_t_it = np.zeros((2, 133))
    Mq_t_it = np.zeros((2, 133))
    
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_b_it[:,i] = (net.res_bus.vm_pu[1:] - vm_orig)/0.0001
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_b_it[:,i] = (net.res_bus.vm_pu[1:] - vm_orig)/0.0001
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001
            
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_l_it[:,i] = (net.res_line.loading_percent/100 - line_orig)/0.0001
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_l_it[:,i] = (net.res_line.loading_percent/100 - line_orig)/0.0001
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001
            
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_t_it[:,i] = (net.res_trafo.loading_percent/100 - trafo_orig)/0.0001
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_t_it[:,i] = (net.res_trafo.loading_percent/100 - trafo_orig)/0.0001
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001

    savetxt('Mp_b_it.csv', Mp_b_it, delimiter=',')
    savetxt('Mq_b_it.csv', Mq_b_it, delimiter=',')
    savetxt('Mp_l_it.csv', Mp_l_it, delimiter=',')
    savetxt('Mq_l_it.csv', Mq_l_it, delimiter=',')
    savetxt('Mp_t_it.csv', Mp_t_it, delimiter=',')
    savetxt('Mq_t_it.csv', Mq_t_it, delimiter=',')
    
    net.sgen.p_mw /= 5
    net.load.p_mw /= 0.1
    net.load.q_mvar /= 0.1
    
    # plots of the sensitivity matrices:
    # plot1 = sns.heatmap(Mp, cbar=False)
    # plt.savefig("P_Sens_matrix_it.svg", format='svg')
    # plot2 = sns.heatmap(Mq, cbar=False)
    # plt.savefig("Q_Sens_matrix_it.svg", format='svg')
    return
    
def sens_matrix_noise(net, std_noise_bus, std_noise_lines, std_noise_trafo):
    net.sgen.p_mw *= 5
    net.load.p_mw *= 0.1
    net.load.q_mvar *= 0.1
      
    pp.runpp(net)
    vm_orig = net.res_bus.vm_pu[1:]
    line_orig = net.res_line.loading_percent[:]/100
    trafo_orig = net.res_trafo.loading_percent[:]/100    
    
    Mp_b_it = np.zeros((135, 133))
    Mq_b_it = np.zeros((135, 133))
    Mp_l_it = np.zeros((136, 133))
    Mq_l_it = np.zeros((136, 133))
    Mp_t_it = np.zeros((2, 133))
    Mq_t_it = np.zeros((2, 133))
    
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_b_it[:,i] = np.random.normal(((net.res_bus.vm_pu[1:] - vm_orig)/0.0001), std_noise_bus)
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_b_it[:,i] = np.random.normal(((net.res_bus.vm_pu[1:] - vm_orig)/0.0001), std_noise_bus)
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001
            
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_l_it[:,i] = np.random.normal(((net.res_line.loading_percent/100 - line_orig)/0.0001), std_noise_lines)
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_l_it[:,i] = np.random.normal(((net.res_line.loading_percent/100 - line_orig)/0.0001), std_noise_lines)
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001
            
    for i in range(133):
            net.sgen.p_mw[i] = net.sgen.p_mw[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mp_t_it[:,i] = np.random.normal(((net.res_trafo.loading_percent/100 - trafo_orig)/0.0001), std_noise_trafo)
            net.sgen.p_mw[i] = net.sgen.p_mw[i] - 0.0001
    
    for i in range(133):
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] + 0.0001 #increase lower limit to increase value of bus voltage
            pp.runpp(net)
            Mq_t_it[:,i] = np.random.normal(((net.res_trafo.loading_percent/100 - trafo_orig)/0.0001), std_noise_trafo)
            net.sgen.q_mvar[i] = net.sgen.q_mvar[i] - 0.0001
            

    savetxt('Mp_b_noise_{}.csv'.format(std_noise_bus*1000), Mp_b_it, delimiter=',')
    savetxt('Mq_b_noise_{}.csv'.format(std_noise_bus*1000), Mq_b_it, delimiter=',')
    savetxt('Mp_l_noise_{}.csv'.format(std_noise_lines*1000), Mp_l_it, delimiter=',')
    savetxt('Mq_l_noise_{}.csv'.format(std_noise_lines*1000), Mq_l_it, delimiter=',')
    savetxt('Mp_t_noise_{}.csv'.format(std_noise_trafo*1000), Mp_t_it, delimiter=',')
    savetxt('Mq_t_noise_{}.csv'.format(std_noise_trafo*1000), Mq_t_it, delimiter=',')
    
    net.sgen.p_mw /= 5
    net.load.p_mw /= 0.1
    net.load.q_mvar /= 0.1
    return
    
    