import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.pyplot as plt
import os
import numba
# from numba import jit

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot
import simbench as sb
import seaborn as sns

import time
import gurobipy as gp
from gurobipy import GRB, quicksum

def make_model(Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, V_prev, line_prev, V_uplim, V_lowlim, sgen_maxP, S_max, trafo_lim, trafo_prev, Q_up, Q_low): 
    m = gp.Model("projection")
    p = m.addVars(len(P_opt), lb=0, ub=GRB.INFINITY)
    q = m.addVars(len(Q_opt), lb=Q_low, ub=Q_up)
    
    for b in range(len(V_prev)): #voltage constraints on buses:
        m.addConstr(gp.quicksum(Mp_b[b,g]*p[g] + Mq_b[b,g]*q[g] for g in range(len(P_prev))) <= V_uplim[V_uplim.index[b]] + gp.quicksum(Mp_b[b,g]*P_prev[g] + Mq_b[b,g]*Q_prev[g] for g in range(len(P_prev))) - V_prev[b], name='V_up{}'.format(b))
        m.addConstr(gp.quicksum(Mp_b[b,g]*p[g] + Mq_b[b,g]*q[g] for g in range(len(P_prev))) >= V_lowlim[V_uplim.index[b]] + gp.quicksum(Mp_b[b,g]*P_prev[g] + Mq_b[b,g]*Q_prev[g] for g in range(len(P_prev))) - V_prev[b], name='V_low{}'.format(b))
    for l in range(len(line_prev)): #current constraints on lines:
        m.addConstr(gp.quicksum(Mp_l[l,g]*p[g] + Mq_l[l,g]*q[g] for g in range(len(P_prev))) <= 1 + gp.quicksum(Mp_l[l,g]*P_prev[g] + Mq_l[l,g]*Q_prev[g] for g in range(len(P_prev))) - line_prev[l], name='line{}'.format(l)) 
    for t in range(2): #transformer limits:
        m.addConstr(gp.quicksum(Mp_t[t,g]*p[g] + Mq_t[t,g]*q[g] for g in range(len(P_prev))) <= trafo_lim[t] + gp.quicksum(Mp_t[t,g]*P_prev[g] + Mq_t[t,g]*Q_prev[g] for g in range(len(P_prev))) - trafo_prev[t], name='trafo{}'.format(t)) 
    for g in range(len(P_prev)): #loop for generators
        #generated power constaint:
        m.addConstr(p[g] <= sgen_maxP[g], name='maxP{}'.format(g))
        #inverter limits:
        m.addQConstr(p[g]*p[g]+q[g]*q[g]<=S_max[g]**2, name='Smax{}'.format(g)) 
    m.setObjective(gp.quicksum((p[g]-P_opt[g])*(p[g]-P_opt[g]) + (q[g]-Q_opt[g])*(q[g]-Q_opt[g]) for g in range(len(P_prev))))
    m.Params.LogToConsole = 0 #for debugging 1 logs, 0 does not 
    m.update()

    return m, p, q

def euclid_opt(m, p, q, Mp_b, Mq_b, Mp_l, Mq_l, Mp_t, Mq_t, P_opt, Q_opt, P_prev, Q_prev, V_prev, line_prev, V_uplim, V_lowlim, sgen_maxP, S_max, trafo_lim, trafo_prev): #X[0] is p, x[1] is q
    for b in range(len(V_prev)): #voltage constraints on buses:   
        m.setAttr("RHS", m.getConstrByName('V_up{}'.format(b)), (V_uplim[V_uplim.index[b]] + np.sum(Mp_b[b,g]*P_prev[g] + Mq_b[b,g]*Q_prev[g] for g in range(len(P_prev))) - V_prev[b]))
        m.setAttr("RHS", m.getConstrByName('V_low{}'.format(b)), (V_lowlim[V_lowlim.index[b]] + np.sum(Mp_b[b,g]*P_prev[g] + Mq_b[b,g]*Q_prev[g] for g in range(len(P_prev))) - V_prev[b]))
    for l in range(len(line_prev)): #current constraints on lines:
        m.setAttr("RHS", m.getConstrByName('line{}'.format(l)), (1 + np.sum(Mp_l[l,g]*P_prev[g] + Mq_l[l,g]*Q_prev[g] for g in range(len(P_prev))) - line_prev[l]))
    for t in range(2): #transformer limits:
        m.setAttr("RHS", m.getConstrByName('trafo{}'.format(t)), (trafo_lim[t] + np.sum(Mp_t[t,g]*P_prev[g] + Mq_t[t,g]*Q_prev[g] for g in range(len(P_prev))) - trafo_prev[t])) 
    for g in range(len(P_prev)): #loop for generators
        #generated power constaint:
        m.setAttr("RHS", m.getConstrByName('maxP{}'.format(g)), sgen_maxP[g])
        #inverter limits:
        m.setAttr("QCRHS", m.getQConstrs()[g], S_max[g])
    m.setObjective(gp.quicksum((p[g]-P_opt[g])*(p[g]-P_opt[g]) + (q[g]-Q_opt[g])*(q[g]-Q_opt[g]) for g in range(len(P_prev))))
    m.update()
    m.Params.LogToConsole = 0 #for debugging 1 logs, 0 does not        
    m.optimize()
    
    if m.status == 2:
        return m, np.array([p[i].X for i in range(len(P_opt))]), np.array([q[i].X for i in range(len(P_opt))])
    else:
        print("no optimal solution found") #we go to relaxed optimization
        copy = m.copy(); 
        copy.feasRelaxS(relaxobjtype=1, minrelax=False, vrelax=False, crelax=True)
        copy.optimize()
        all_vars = copy.getVars()
        values = copy.getAttr("X", all_vars)
        return m, np.array(values[0:133]), np.array(values[133:266])
