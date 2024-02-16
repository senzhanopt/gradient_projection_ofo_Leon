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
from datetime import datetime, timedelta
import matplotlib.dates as md

import simbench as sb
import seaborn as sns

import time
import gurobipy as gp
from gurobipy import GRB, quicksum

from tqdm import tqdm

# %% importing/processing data:
    
Data_dict = {}
sgen_restot_dict = {}
restot = 0

#static:
delay = [0]
alpha = [2, 5, 10, 20, 50]
cs = 100

for d in delay:
    for a in alpha:
        key = 'static_'+str(d)+'-'+str(a)
        Data_dict[key] = pd.read_csv('results_'+key+'.csv')
        sgen_restot_dict[key] = pd.Series(float(0), range(cs))
        for t in range(cs):
            for g in range(133):
                restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][t]
            sgen_restot_dict[key][t] = restot
            restot = 0
            
delay = [1,2,3]
alpha = [5]

for d in delay:
    for a in alpha:
        key = 'static_'+str(d)+'-'+str(a)
        Data_dict[key] = pd.read_csv('results_'+key+'.csv')
        sgen_restot_dict[key] = pd.Series(float(0), range(cs))
        for t in range(cs):
            for g in range(133):
                restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][t]
            sgen_restot_dict[key][t] = restot
            restot = 0
            
delay = [2]
alpha = [1, 2]

for d in delay:
    for a in alpha:
        key = 'static_'+str(d)+'-'+str(a)
        Data_dict[key] = pd.read_csv('results_'+key+'.csv')
        sgen_restot_dict[key] = pd.Series(float(0), range(cs))
        for t in range(cs):
            for g in range(133):
                restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][t]
            sgen_restot_dict[key][t] = restot
            restot = 0
         
delay = [0]
alpha = [5]
noise_std_voltage = [1, 5]
noise_std_current = [1, 5]
noise_std_trafo = [1, 5]

for d in delay:
    for a in alpha:
        for nv in noise_std_voltage:
            for nc in noise_std_current:
                for nt in noise_std_trafo:
                    key = 'static_{}-{}-noise-{}-{}-{}'.format(d,a,nv,nc,nt)
                    Data_dict[key] = pd.read_csv('results_'+key+'.csv')
                    sgen_restot_dict[key] = pd.Series(float(0), range(cs))
                    for t in range(cs):
                        for g in range(133):
                            restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][t]
                        sgen_restot_dict[key][t] = restot
                        restot = 0


#dynamic:
alpha = [1,2,5,10,20,50]
steps = 96
cs = 2

for a in alpha:
    key = 'dyn_'+str(d)+'-'+str(a)+'-'+str(cs)
    Data_dict[key] = pd.read_csv('results_'+key+'.csv')
    sgen_restot_dict[key] = pd.Series(float(0), range(steps))
    for s in range(steps):
        for g in range(133):
            restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][s]
        sgen_restot_dict[key][s] = restot
        restot = 0
            
alpha = [10]
steps = 96
cs = [1,5,15]

for a in alpha:
    for step in cs:
        key = 'dyn_'+str(d)+'-'+str(a)+'-'+str(step)
        Data_dict[key] = pd.read_csv('results_'+key+'.csv')
        sgen_restot_dict[key] = pd.Series(float(0), range(steps))
        for s in range(steps):
            for g in range(133):
                restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][s]
            sgen_restot_dict[key][s] = restot
            restot = 0
                
alpha = [1, 5]
delay = [0,1]
cs = [30,2]
steps = 96

for d in delay:
    for a in alpha:
        for step in cs:
            key = 'dyn_'+str(d)+'-'+str(a)+'-'+str(step)
            Data_dict[key] = pd.read_csv('results_'+key+'.csv')
            sgen_restot_dict[key] = pd.Series(float(0), range(steps))
            for s in range(steps):
                for g in range(133):
                    restot = restot + Data_dict[key]['PV{}_p_mw_grad'.format(g)][s]
                sgen_restot_dict[key][s] = restot
                restot = 0


#no curtailment:
cs = 100
restot = 0
key = 'static_no_curt'
sgen_restot_dict[key] = pd.Series(float(0), range(cs))
for t in range(cs):
    for g in range(133):
        restot = restot + Data_dict['static_0-2']['PV{}_p_mw'.format(g)][t]
    sgen_restot_dict['static_no_curt'][t] = restot
    restot = 0
    
steps = 96
key = 'dyn_no_curt'
sgen_restot_dict[key] = pd.Series(float(0), range(steps))
for s in range(steps):
    for g in range(133):
        restot = restot + Data_dict['dyn_0-1-2']['PV{}_p_mw'.format(g)][s]
    sgen_restot_dict['dyn_no_curt'][s] = restot
    restot = 0
            
# %% plotting data
    
#problematic line 67
fig4, ax1 = plt.subplots(figsize=(6,3))
Data_dict['static_0-2']['line66_percent'].plot(ax=ax1, use_index=False, label='no curtailment')
Data_dict['static_0-2']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.02')
Data_dict['static_0-5']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.05')
Data_dict['static_0-10']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.1')
Data_dict['static_0-20']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.2')
Data_dict['static_0-50']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.5')
plt.xlabel('controller time-steps [-]')
plt.ylabel('loading [%]')
# plt.title("Transient response line 67 loading")
ax = plt.gca()
ax.set_xlim([-2, 60])
plt.legend(loc="right")
plt.savefig('Current_line67.png', format='png', dpi=300, bbox_inches='tight')
 
#problematic bus: 76, note bus numbers do not coincide due to the bus index and title not corresponding
fig5, ax1 = plt.subplots(figsize=(6,3))
Data_dict['static_0-2']['bus68_vm_pu'].plot(ax=ax1, use_index=False, label='no curtailment')
Data_dict['static_0-2']['bus68_vm_pu_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.02')
Data_dict['static_0-5']['bus68_vm_pu_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.05')
Data_dict['static_0-10']['bus68_vm_pu_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.1')
Data_dict['static_0-20']['bus68_vm_pu_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.2')
Data_dict['static_0-50']['bus68_vm_pu_grad'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.5')
plt.xlabel('controller time-steps [-]')
plt.ylabel('Voltage [p.u.]')
# plt.title("Transient response voltage bus 76")
ax = plt.gca()
ax.set_xlim([-2, 60])
plt.legend(loc="lower right")
plt.savefig('Voltage_bus76.png', format='png', dpi=300, bbox_inches='tight')

fig6, ax1 = plt.subplots(figsize=(6,3))
sgen_restot_dict['dyn_no_curt'].plot(ax=ax1, use_index=False, label='no curtailment')
sgen_restot_dict['dyn_0-10-1'].plot(ax=ax1, use_index=False, label='cs = 1')
sgen_restot_dict['dyn_0-10-2'].plot(ax=ax1, use_index=False, label='cs = 2')
sgen_restot_dict['dyn_0-10-5'].plot(ax=ax1, use_index=False, label='cs = 5')
sgen_restot_dict['dyn_0-10-15'].plot(ax=ax1, use_index=False, label='cs = 15')
plt.xticks([20,24,28,32,36,40,44,48], ["5:00","06:00","07:00", "08:00","09:00","10:00", "11:00","12:00"], rotation = 90)
ax = plt.gca()
ax.set_xlim([18, 50])
ax.set_ylim([0, 150])
# plt.title("Dynamic response, total PV generation varying amount of cs")
plt.xlabel('time [hours]')
plt.ylabel('Active power [MW]')
plt.legend()
plt.savefig('Dynamic_control_steps.png', format='png', dpi=300, bbox_inches='tight')

fig7, ax1 = plt.subplots(figsize=(6,3))
sgen_restot_dict['dyn_no_curt'].plot(ax=ax1, use_index=False, label='no curtailment')
sgen_restot_dict['dyn_0-50-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.5')
sgen_restot_dict['dyn_0-20-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.2')
sgen_restot_dict['dyn_0-10-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.1')
sgen_restot_dict['dyn_0-5-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.05')
sgen_restot_dict['dyn_0-2-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.02')
sgen_restot_dict['dyn_0-1-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.01')
plt.xticks([20,24,28,32,36,40,44,48], ["5:00","06:00","07:00", "08:00","09:00","10:00", "11:00","12:00"], rotation = 45)
ax = plt.gca()
ax.set_xlim([18, 50])
ax.set_ylim([0, 150])
# plt.title(r"Dynamic response, total PV generation varying amount of $\alpha$")
plt.xlabel('time [hours]')
plt.ylabel('Active power [MW]')
plt.legend(loc="upper right")
plt.savefig('Dynamic_alpha.png', format='png', dpi=300, bbox_inches='tight')

#varying delay:
fig8, ax1 = plt.subplots(figsize=(6,3))
sgen_restot_dict['static_no_curt'].plot(ax=ax1, use_index=False, label='no curtailment')
sgen_restot_dict['static_0-5'].plot(ax=ax1, use_index=False, label='delay = 0')
sgen_restot_dict['static_1-5'].plot(ax=ax1, use_index=False, label='delay = 1')
sgen_restot_dict['static_2-5'].plot(ax=ax1, use_index=False, label='delay = 2')
sgen_restot_dict['static_3-5'].plot(ax=ax1, use_index=False, label='delay = 3')
# plt.title(r"Transient respose, total PV generation, $\alpha$ = 0.05")
plt.xlabel('controller time-steps [-]')
plt.ylabel('Active power [MW]')
plt.legend(loc="lower right")
plt.savefig('totalPV_varyingdelay.png', format='png', dpi=300, bbox_inches='tight')


fig9, ax1 = plt.subplots(figsize=(6,3))
sgen_restot_dict['dyn_no_curt'].plot(ax=ax1, use_index=False, label='no curtailment')
sgen_restot_dict['dyn_0-1-30'].plot(ax=ax1, use_index=False, label=r'cs = 30, $\alpha$ = 0.01, delay = 0')
sgen_restot_dict['dyn_1-1-30'].plot(ax=ax1, use_index=False, label=r'cs = 30, $\alpha$ = 0.01, delay = 1')
sgen_restot_dict['dyn_1-5-2'].plot(ax=ax1, use_index=False, label=r'cs = 2, $\alpha$ = 0.05, delay = 1')
plt.xticks([20,24,28,32,36,40,44,48,52,56], ["5:00","06:00","07:00", "08:00","09:00","10:00", "11:00","12:00", "13:00", "14:00"], rotation = 90)
ax = plt.gca()
ax.set_xlim([18, 56])
ax.set_ylim([20, 160])
plt.xlabel('time [hours]')
plt.ylabel('Active power [MW]')
plt.legend(loc="upper right")
plt.savefig('totalPV_dynamic_varyingdelay.png', format='png', dpi=300, bbox_inches='tight')

#varying stepsize, fixed delay:
fig10, ax1 = plt.subplots(figsize=(6,3))
sgen_restot_dict['static_no_curt'].plot(ax=ax1, use_index=False, label='no curtailment')
sgen_restot_dict['static_2-1'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.01')
sgen_restot_dict['static_2-2'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.02')
sgen_restot_dict['static_2-5'].plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.05')
# sgen_restot_0_2_1.plot(ax=ax1, use_index=False, label=r'$\alpha$ = 0.1')
# plt.title("Total PV generation, delay = 2")
plt.xlabel('controller time-steps [-]')
plt.ylabel('Active power [MW]')
plt.legend()
plt.savefig('totalPV_varyingalpha.png', format='png', dpi=300, bbox_inches='tight')

#measurement noise, problematic line: 67
fig6, ax1 = plt.subplots(figsize=(6,4))
Data_dict['static_0-5']['line66_percent_grad'].plot(ax=ax1, use_index=False, label='no noise')
Data_dict['static_0-5-noise-1-1-1']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\sigma_b$ = 0.001, $\sigma_l$ = 0.001, $\sigma_T$ = 0.001')
Data_dict['static_0-5-noise-5-1-1']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\sigma_b$ = 0.005, $\sigma_l$ = 0.001, $\sigma_T$ = 0.001')
Data_dict['static_0-5-noise-1-5-1']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\sigma_b$ = 0.001, $\sigma_l$ = 0.005, $\sigma_T$ = 0.001')
Data_dict['static_0-5-noise-1-1-5']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\sigma_b$ = 0.001, $\sigma_l$ = 0.001, $\sigma_T$ = 0.005')
Data_dict['static_0-5-noise-5-5-5']['line66_percent_grad'].plot(ax=ax1, use_index=False, label=r'$\sigma_b$ = $\sigma_l$ = $\sigma_T$ = 0.005')
plt.xlabel('controller time-steps [-]')
plt.ylabel('loading [%]')
ax = plt.gca()
ax.set_xlim([5, 45])
ax.set_ylim([97, 107])
# plt.title(r"line 67 loading, $\alpha$ = 0.05")
plt.legend()
plt.savefig('Meas_noise.png', format='png', dpi=300, bbox_inches='tight')




