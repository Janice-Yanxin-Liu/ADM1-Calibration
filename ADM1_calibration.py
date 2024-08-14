import numpy as np
import pandas as pd
import sunode
import matplotlib.pyplot as plt
import sunode.wrappers.as_aesara
import pymc3 as pm

# ------------------- Experimental Observations for calibration ---------------- #
times = np.linspace(0,312,313)

obs = pd.read_csv('experimental_observations.csv', header = 0, index_col = None)

index_biogas = np.array([0,1,2,3,4,5,6,7,8,9,10])
index_TAN = np.array([6, 14, 20, 27, 43, 49])
index_pH = np.array([6, 18, 44, 50, 57])

biogas_obs = obs['daily_biogas (m3/d)'].to_numpy()[index_biogas]

pH_obs = obs['pH'].to_numpy()[index_pH]

TAN_obs = obs['IN (kmolN/m3)'].to_numpy()[index_TAN]

# ------------------------------ Model Input ---------------------------------- # 
q = 2.06e-4
V_liq = 0.0052
V_gas = 0.00322
    
# Inflow concentrations
S_suf = 6.542
S_aaf = 0
S_faf = 0
S_vaf = 0
S_buf = 0
S_prof = 0
S_acf = 23.2556
S_h2f = 0
S_ch4f = 0
S_ICf = 0.1049
S_INf = 0.0384
S_If = 1.0495
X_cf = 0
X_chf = 49.3911
X_prf = 26.5423
X_lif = 0
X_suf = 0
X_aaf = 0
X_faf = 0
X_c4f = 0
X_prof =0
X_acf = 0
X_h2f = 0 
X_If = 9.0461
S_catf = 0.0631
S_anf = 0.1423
    
# Fractions
f_sI_xc = 0.0122
f_xI_xc = 0.1052
f_ch_xc = 0.574
f_pr_xc = 0.3085
f_li_xc = 0.0001
    
f_fa_li = 0.95
f_h2_su = 0.19
f_bu_su = 0.13
f_pro_su = 0.27
f_ac_su = 0.41
f_h2_aa = 0.06
f_va_aa = 0.23
f_bu_aa = 0.26
f_pro_aa = 0.05
f_ac_aa = 0.4
    
# Carbon and Nitrogen contents
C_xc = 0.021
C_sI = 0.03 
C_ch = 0.0236
C_pr = 0.0127
C_li = 0.022
C_xI = 0.03
C_su = 0.0313
C_aa = 0.03
C_fa = 0.0217
C_va = 0.024
C_bu = 0.025
C_pro = 0.0268
C_ac = 0.0313
C_bac = 0.0313
C_ch4 = 0.0156
N_xc = 0.00266
N_I = 0.004285714
N_aa = 0.007
N_bac = 0.005714286
    
# Yield Uptake Components
Y_su = 0.1
Y_aa = 0.08
Y_fa = 0.06
Y_c4 = 0.06
Y_pro = 0.04
Y_ac = 0.05
Y_h2 = 0.06
    
# Disintegration, hydrolysis, decay rates, maximum uptake rates
k_hyd_li = 3
k_m_su = 30
k_m_aa = 50
k_m_fa = 6
k_m_c4 = 20
k_m_pro = 13
k_m_ac = 10
k_m_h2 = 35
k_dec_Xsu = 0.02
k_dec_Xaa = 0.02
k_dec_Xfa = 0.02
k_dec_Xc4 = 0.02
k_dec_Xpro = 0.02
k_dec_Xac = 0.02
k_dec_Xh2 = 0.02

# Half Saturation coefficient, 50% inhibitory coefficient
K_S_IN = 0.0001
K_S_su = 0.5
K_S_aa = 0.3
K_S_fa = 0.4
K_S_c4 = 0.2
K_S_pro = 0.1
K_S_ac = 0.15
K_S_h2 = 7e-6
K_I_nh3 = 0.01 
K_Ih2_fa = 5e-6
K_Ih2_c4 = 1e-5
K_Ih2_pro = 3.5e-6
    
# Acid and Gas Parameters
kLa = 200
K_H_h2o_base = 0.0313
k_P = 50000
P_atm = 1.013
T_base = 298.15
T_op = 312.65
R = 0.083145
    
k_A_Bva = 1e10
k_A_Bbu = 1e10
k_A_Bpro = 1e10
k_A_Bac = 1e10
k_A_Bco2 = 1e10
k_A_BIN = 1e10
    
pH_UL_h2 = 6
pH_LL_h2 = 5
pH_UL_aa = 5.5
pH_LL_aa = 4
pH_UL_ac = 7
pH_LL_ac = 6
    
pHLim_aa = 10**(-(pH_UL_aa + pH_LL_aa)/2.0)
pHLim_ac = 10**(-(pH_UL_ac + pH_LL_ac)/2.0)
pHLim_h2 = 10**(-(pH_UL_h2 + pH_LL_h2)/2.0)
    
k_aa = 3.0/(pH_UL_aa-pH_LL_aa)
k_ac = 3.0/(pH_UL_ac-pH_LL_ac)
k_h2 = 3.0/(pH_UL_h2-pH_LL_h2)
    
K_a_va = 10**-4.68
K_a_bu = 10**-4.82
K_a_pro = 10**-4.88
K_a_ac = 10**-4.76

K_a_co2 = 10**-6.35 * np.exp(7646/R/100*(1/T_base-1/T_op))
K_a_IN = 10**-9.25 * np.exp(51965/R/100*(1/T_base-1/T_op))

K_w = 10**(-14) * np.exp(55900/100/R*(1/T_base-1/T_op))

K_H_co2 = 0.035 * np.exp(-19410/R/100*(1/T_base-1/T_op))
K_H_ch4 = 0.0014 * np.exp(-14240/R/100*(1/T_base-1/T_op))
K_H_h2 = 7.8e-4 * np.exp(-4180/R/100*(1/T_base-1/T_op))

## ------------------------ Model Construction -----------------------------##

def ADM1(t, y, p):
  
    # ~~~~~~~~~~~~ Acid-base association/dissociation rates ~~~~~~~~~~~~~~~~~~
    r_A_4 = k_A_Bva*(y.y26*(K_a_va + y.y35) - K_a_va * y.y3)
    r_A_5 = k_A_Bbu*(y.y27*(K_a_bu + y.y35) - K_a_bu * y.y4)
    r_A_6 = k_A_Bpro*(y.y28*(K_a_pro + y.y35)- K_a_pro * y.y5)
    r_A_7 = k_A_Bac*(y.y29*(K_a_ac + y.y35) - K_a_ac * y.y6)

    r_A_10 = k_A_Bco2*(y.y30*(K_a_co2 + y.y35) - K_a_co2*y.y9); # IC
    r_A_11 = k_A_BIN*(y.y31*(K_a_IN + y.y35) - K_a_IN*y.y10); # IN

    # ~~~~~~~~~~~ Gas transfer rate from liquid to gas phase ~~~~~~~~~~~~~~
    r_T_8 = kLa*(y.y7 - 16*K_H_h2*(y.y32*R*T_op/16.0))
    r_T_9 = kLa*(y.y8 - 64*K_H_ch4*(y.y33*R*T_op/64.0))
    r_T_10 = kLa*(y.y9 - y.y30 - K_H_co2*(y.y34*R*T_op))
    
    # ~~~~~~~~~~~ C balance Stoich(i) calculation ~~~~~~~~~~~~~~~~~~~~~~~~~
    stoich1 = -C_xc+f_sI_xc*C_sI+f_ch_xc*C_ch+f_pr_xc*C_pr+f_li_xc*C_li+f_xI_xc*C_xI
    stoich2 = -C_ch+C_su
    stoich3 = -C_pr+C_aa
    stoich4 = -C_li+(1.0-f_fa_li)*C_su+f_fa_li*C_fa
    stoich5 = -C_su+(1.0-Y_su)*(f_bu_su*C_bu+f_pro_su*C_pro+f_ac_su*C_ac)+Y_su*C_bac
    stoich6 = -C_aa+(1.0-Y_aa)*(f_va_aa*C_va+f_bu_aa*C_bu+f_pro_aa*C_pro+f_ac_aa*C_ac)+Y_aa*C_bac
    stoich7 = -C_fa+(1.0-Y_fa)*0.7*C_ac+Y_fa*C_bac
    stoich8 = -C_va+(1.0-Y_c4)*0.54*C_pro+(1.0-Y_c4)*0.31*C_ac+Y_c4*C_bac
    stoich9 = -C_bu+(1.0-Y_c4)*0.8*C_ac+Y_c4*C_bac
    stoich10 = -C_pro+(1.0-Y_pro)*0.57*C_ac+Y_pro*C_bac
    stoich11 = -C_ac+(1.0-Y_ac)*C_ch4+Y_ac*C_bac
    stoich12 = (1.0-Y_h2)*C_ch4+Y_h2*C_bac
    stoich13 = -C_bac+C_xc
    
    # ~~~~~~~~~~~~~~~~~~~~ Inhibition calculation ~~~~~~~~~~~~~~~~~~~~~~~~~
    I_pH_aa = pHLim_aa**k_aa/(y.y35**k_aa + pHLim_aa**k_aa)
    I_pH_ac = pHLim_ac**k_ac /(y.y35**k_ac + pHLim_ac**k_ac)
    I_pH_h2 = pHLim_h2**k_h2 /(y.y35**k_h2 + pHLim_h2**k_h2)
    I_IN_lim = 1/(1+K_S_IN/y.y10)
    I_nh3 = 1/(1+y.y31/K_I_nh3)
    I_h2_fa = 1.0/(1.0+y.y7/K_Ih2_fa)
    I_h2_c4 = 1.0/(1.0+y.y7/K_Ih2_c4)
    I_h2_pro = 1.0/(1.0+y.y7/K_Ih2_pro)
    
    inhib56 = I_pH_aa*I_IN_lim
    inhib7 = inhib56*I_h2_fa
    inhib89 = inhib56*I_h2_c4
    inhib10 = inhib56*I_h2_pro
    inhib11 = I_pH_ac*I_IN_lim*I_nh3
    inhib12 = I_pH_h2*I_IN_lim
    
    # ~~~~~~~~~~~Calculate reaction rates ro(1-19) of processes ~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ro1 = p.k_dis*y.y12
    ro2 = p.k_hyd_ch*y.y13
    ro3 = p.k_hyd_pr*y.y14
    ro4 = k_hyd_li*y.y15
    ro5 = k_m_su*y.y0/(K_S_su+y.y0)*y.y16*inhib56
    ro6 = k_m_aa*y.y1/(K_S_aa+y.y1)*y.y17*inhib56
    ro7 = k_m_fa*y.y2/(K_S_fa+y.y2)*y.y18*inhib7
    ro8 = k_m_c4*y.y3/(K_S_c4+y.y3)*y.y19*y.y3/(y.y4+y.y3+1e-6)*inhib89
    ro9 = k_m_c4*y.y4/(K_S_c4+y.y4)*y.y19*y.y4/(y.y3+y.y4+1e-6)*inhib89
    ro10 = k_m_pro*y.y5/(K_S_pro+y.y5)*y.y20*inhib10
    ro11 = k_m_ac*y.y6/(K_S_ac+y.y6)*y.y21*inhib11
    ro12 = k_m_h2*y.y7/(K_S_h2+y.y7)*y.y22*inhib12
    ro13 = k_dec_Xsu*y.y16
    ro14 = k_dec_Xaa*y.y17
    ro15 = k_dec_Xfa*y.y18
    ro16 = k_dec_Xc4*y.y19
    ro17 = k_dec_Xpro*y.y20
    ro18 = k_dec_Xac*y.y21
    ro19 = k_dec_Xh2*y.y22
    
    # ~~~~~~~~~~~~~~~~ Gas flow calculation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p_gas_h2 = y.y32*R*T_op/16.0   # p_gas_h2
    p_gas_ch4 = y.y33*R*T_op/64.0  # p_gas_ch4
    p_gas_co2 = y.y34*R*T_op       # p_gas_co2
    
    p_gas_h2o = K_H_h2o_base * np.exp(5290.0*(1.0/T_base - 1.0/T_op)) # T adjustement for water vapour saturation pressure
    P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    q_gas = k_P*(P_gas-P_atm)*P_gas/P_atm
        
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~pH (S_h_ion)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reference: Thamsiriroj and Murphy, 2011
    A1 = (q/V_liq)*(S_anf - y.y25)
    
    A2 = (q/V_liq)*(S_INf - y.y10) - Y_su*N_bac*ro5 + (N_aa - Y_aa*N_bac)*ro6 - Y_fa*N_bac*ro7\
    -Y_c4*N_bac*ro8 - Y_c4*N_bac*ro9 - Y_pro*N_bac*ro10 - Y_ac*N_bac*ro11 - Y_h2*N_bac*ro12\
    + (N_bac - N_xc)*(ro13+ro14+ro15+ro16+ro17+ro18+ro19)+(N_xc - f_xI_xc*N_I- f_sI_xc*N_I - f_pr_xc*N_aa)*ro1
    
    A3 = (q/V_liq)*(S_ICf - y.y9) - (stoich1*ro1 + stoich2*ro2 + stoich3*ro3 + stoich4*ro4\
    + stoich5*ro5 + stoich6*ro6 + stoich7*ro7 + stoich8*ro8 + stoich9*ro9 + stoich10*ro10\
    + stoich11*ro11 + stoich12*ro12 + stoich13*(ro13+ro14+ro15+ro16+ro17+ ro18 + ro19)) - r_T_10
    
    A4 = (q/V_liq)*(S_acf - y.y6) + (1-Y_su)*f_ac_su*ro5 + (1-Y_aa)*f_ac_aa*ro6 + (1-Y_fa)*0.7*ro7\
    + (1-Y_c4)*0.31*ro8 + (1-Y_c4)*0.8*ro9 + (1-Y_pro)*0.57*ro10 - ro11
    
    A5 = (q/V_liq)*(S_prof - y.y5) + (1-Y_su)*f_pro_su*ro5 + (1-Y_aa)*f_pro_aa*ro6+(1-Y_c4)*0.54*ro8 - ro10
    
    A6 = (q/V_liq)*(S_buf - y.y4) + (1-Y_su)*f_bu_su*ro5 + (1-Y_aa)*f_bu_aa*ro6 - ro9
    
    A7 = (q/V_liq)*(S_vaf - y.y3) + (1-Y_aa)*f_va_aa*ro6 - ro8
    
    A8 = (q/V_liq)*(S_catf - y.y24)

    A = A1 + A2*K_a_IN/(K_a_IN+y.y35) + A3*K_a_co2/(K_a_co2+y.y35) + (1/64)*A4*K_a_ac/(K_a_ac+y.y35)\
    + (1/112)*A5*K_a_pro/(K_a_pro+y.y35) + (1/160)*A6*K_a_bu/(K_a_bu+y.y35)\
    + (1/208)*A7*K_a_va/(K_a_va+y.y35) - A2 - A8

    B = 1 + y.y10*K_a_IN/((K_a_IN+y.y35)**2) + y.y9*K_a_co2/((K_a_co2+y.y35)**2)\
    + (1/64)*y.y6*K_a_ac/((K_a_ac+y.y35)**2) + (1/112)*y.y5*K_a_pro/((K_a_pro+y.y35)**2)\
    + (1/160)*y.y4*K_a_bu/((K_a_bu+y.y35)**2) + (1/208)*y.y3*K_a_va/((K_a_va+y.y35)**2) + K_w/(y.y35**2)
        
    return {
        # -----------------------------------------------/
        #                BUILD ODE SYSTEM                /
        # -----------------------------------------------/
        #~~~~~~~~~~~~~ Water phase biochemical reaction~~~~~~~~~~~~~~~~~~~~~~~
        'y0': (q/V_liq)*(S_suf - y.y0) + ro2 +(1-f_fa_li)*ro4 - ro5,
        
        'y1': (q/V_liq)*(S_aaf - y.y1) + ro3 - ro6,
        
        'y2': (q/V_liq)*(S_faf - y.y2) + f_fa_li*ro4 - ro7,
        
        'y3': (q/V_liq)*(S_vaf - y.y3) + (1-Y_aa)*f_va_aa*ro6 - ro8,
        
        'y4': (q/V_liq)*(S_buf - y.y4) + (1-Y_su)*f_bu_su*ro5 + (1-Y_aa)*f_bu_aa*ro6 - ro9,
        
        'y5': (q/V_liq)*(S_prof - y.y5) + (1-Y_su)*f_pro_su*ro5 + (1-Y_aa)*f_pro_aa*ro6+(1-Y_c4)*0.54*ro8 - ro10,
        
        'y6': (q/V_liq)*(S_acf - y.y6) + (1-Y_su)*f_ac_su*ro5 + (1-Y_aa)*f_ac_aa*ro6 + (1-Y_fa)*0.7*ro7\
        + (1-Y_c4)*0.31*ro8 + (1-Y_c4)*0.8*ro9 + (1-Y_pro)*0.57*ro10 - ro11,
        
        'y7': (q/V_liq)*(S_h2f - y.y7) + (1-Y_su)*f_h2_su*ro5 + (1-Y_aa)*f_h2_aa*ro6 + (1-Y_fa)*0.3*ro7\
        + (1-Y_c4)*0.15*ro8 + (1-Y_c4)*0.2*ro9 +(1-Y_pro)*0.43*ro10 - ro12 - r_T_8,
        
        'y8': (q/V_liq)*(S_ch4f - y.y8) + (1-Y_ac)*ro11 + (1-Y_h2)*ro12 - r_T_9,
        
        'y9': (q/V_liq)*(S_ICf - y.y9) - (stoich1*ro1 + stoich2*ro2 + stoich3*ro3 + stoich4*ro4\
        + stoich5*ro5 + stoich6*ro6 + stoich7*ro7 + stoich8*ro8 + stoich9*ro9 + stoich10*ro10\
        + stoich11*ro11 + stoich12*ro12 + stoich13*(ro13+ro14+ro15+ro16+ro17\
        + ro18 + ro19)) - r_T_10,
        
        'y10': (q/V_liq)*(S_INf - y.y10) - Y_su*N_bac*ro5 + (N_aa - Y_aa*N_bac)*ro6 - Y_fa*N_bac*ro7\
        -Y_c4*N_bac*ro8 - Y_c4*N_bac*ro9 - Y_pro*N_bac*ro10 - Y_ac*N_bac*ro11 - Y_h2*N_bac*ro12\
        +(N_bac - N_xc)*(ro13+ro14+ro15+ro16+ro17+ro18+ro19)+(N_xc - f_xI_xc*N_I\
        - f_sI_xc*N_I - f_pr_xc*N_aa)*ro1,
        
        'y11': (q/V_liq)*(S_If - y.y11) + f_sI_xc*ro1,
        # Particulate
        'y12': (q/V_liq)*(X_cf - y.y12)- ro1 + ro13 + ro14 + ro15 + ro16 + ro17 + ro18 + ro19,
        'y13': (q/V_liq)*(X_chf - y.y13) + f_ch_xc*ro1 - ro2,
        'y14': (q/V_liq)*(X_prf - y.y14) + f_pr_xc*ro1 - ro3,
        'y15': (q/V_liq)*(X_lif - y.y15) + f_li_xc*ro1 - ro4,
        'y16': (q/V_liq)*(X_suf - y.y16) + Y_su*ro5 - ro13,
        'y17': (q/V_liq)*(X_aaf - y.y17) + Y_aa*ro6 - ro14,
        'y18': (q/V_liq)*(X_faf - y.y18) + Y_fa*ro7 - ro15,
        'y19': (q/V_liq)*(X_c4f - y.y19) + Y_c4*ro8 + Y_c4*ro9 - ro16,
        'y20': (q/V_liq)*(X_prof - y.y20) + Y_pro*ro10 - ro17,
        'y21': (q/V_liq)*(X_acf - y.y21) + Y_ac*ro11 - ro18,
        'y22': (q/V_liq)*(X_h2f - y.y22) + Y_h2*ro12 - ro19,
        'y23': (q/V_liq)*(X_If - y.y23) + f_xI_xc*ro1,
        # Cation and anion;
        'y24': (q/V_liq)*(S_catf - y.y24),
        'y25': (q/V_liq)*(S_anf - y.y25),
    
    # ~~~~~~~~~~~~~~~~~ Physicochemical Ion states ~~~~~~~~~~~~~~~~~~~~~~~~
        'y26': - r_A_4,
        'y27': - r_A_5,
        'y28': - r_A_6,
        'y29': - r_A_7,
        'y30': - r_A_10,
        'y31': - r_A_11,
    
    # ~~~~~~~~~~~~~~~~~~~~~ Gas phase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'y32': - y.y32*(q_gas/V_gas) +r_T_8*(V_liq/V_gas),
        'y33': - y.y33*(q_gas/V_gas) + r_T_9*(V_liq/V_gas),
        'y34': - y.y34*(q_gas/V_gas)+ r_T_10*(V_liq/V_gas),
        
        'y35': A/B,
        
    }


# --------------------- Bayesian Calibration Framework ------------------------- #
with pm.Model() as model:

    y16_start = pm.Normal('y16_start', mu = 3.571, sigma = 1.482)
    y21_start = pm.Normal('y21_start', mu = 1.785, sigma = 0.741)
    y24_start = pm.Gamma('y24_start', alpha = 1.5, beta = 4.0)
    y25_start = pm.Gamma('y25_start', alpha = 1.5, beta = 5.0)

    k_dis = pm.Normal('k_dis', mu = 3.55, sigma = 1.367)
    k_hyd_ch = pm.Gamma('k_hyd_ch', alpha = 2.0, beta = 4.0)
    k_hyd_pr = pm.Gamma('k_hyd_pr', alpha = 2.0, beta = 4.0)

    y0 = {
        'y0': np.array(0.0277),
        'y1': np.array(0.4640),
        'y2': np.array(0.0163),
        'y3': np.array(0.006),
        'y4': np.array(0.064),
        'y5': np.array(0.0365),
        'y6': np.array(0.121),
        'y7': np.array(2.2e-7),
        'y8': np.array(0.05),
        'y9': np.array(0.03575),
        'y10': np.array(0.18976),
        'y11': np.array(3.21344),
        'y12': np.array(15.0),
        'y13': np.array(7.8983),
        'y14': np.array(2.0441),
        'y15': np.array(5.419),
        'y16': (y16_start, ()),
        'y17': np.array(3.047),
        'y18': np.array(2.284),
        'y19': np.array(2.284),
        'y20': np.array(1.523),
        'y21': (y21_start, ()),
        'y22': np.array(2.284),
        'y23': np.array(3.971),
        'y24': (y24_start, ()),
        'y25': (y25_start, ()),
        'y26': np.array(0.00599),
        'y27': np.array(0.06393),
        'y28': np.array(0.03696),
        'y29': np.array(0.12089),
        'y30': np.array(0.0345),
        'y31': np.array(0.00647),
        'y32': np.array(1.588e-5),
        'y33': np.array(1.30022),
        'y34': np.array(0.01591),
        'y35': np.array(0.00000001585),
    }
    
    params = {
        'k_dis': (k_dis, ()),
        'k_hyd_ch': (k_hyd_ch, ()),
        'k_hyd_pr': (k_hyd_pr, ()),
        'unused_extra': np.array(5.),
    }


    from sunode.wrappers.as_aesara import solve_ivp
    solution, *_ = solve_ivp(
        y0=y0,
        params=params,
        rhs = ADM1,
        # The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )

# Data Likelihood
    
    sd1 = pm.HalfNormal('sd1')
    sd2 = pm.HalfNormal('sd2')
    sd3 = pm.HalfNormal('sd3')
    
    # q_gas
    p_gas_h2 = solution['y32']*R*T_op/16.0   # p_gas_h2
    p_gas_ch4 = solution['y33']*R*T_op/64.0  # p_gas_ch4
    p_gas_co2 = solution['y34']*R*T_op       # p_gas_co2
    
    p_gas_h2o = K_H_h2o_base * np.exp(5290.0*(1.0/T_base - 1.0/T_op)) # T adjustement for water vapour saturation pressure
    P_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o
    q_gas_cal = k_P*(P_gas-P_atm)*P_gas/P_atm
    
    acc_d_biogas = pm.Normal('acc_d_biogas', mu = q_gas_cal[index_biogas], sigma=sd1, observed = biogas_obs)
    
    pH = pm.Normal('pH', mu = -np.log10(solution['y35'][index_pH]), sigma = sd2, observed = pH_obs)
    
    TAN = pm.Normal('TAN', mu = solution['y10'][index_TAN], sigma = sd3, observed = TAN_obs)
    
    # record the biogas, pH and TAN simulated for different combination of parameter sets
    biogas_pred = pm.Deterministic('biogas_pred', q_gas_cal)
    pH_pred = pm.Deterministic('pH_pred', -np.log10(solution['y35']))
    TAN_pred = pm.Deterministic('TAN_pred', solution['y10'])
    tCOD_pred = pm.Deterministic('tCOD_pred', 
                                solution['y1'] + solution['y2'] + solution['y3'] + solution['y4'] + \
                                solution['y5'] + solution['y6'] + solution['y7'] + solution['y8'] + \
                                solution['y11'] + solution['y12'] + solution['y13'] + solution['y14'] + \
                                solution['y15'] + solution['y16'] + solution['y17'] + solution['y18'] + \
                                solution['y19'] + solution['y20'] + solution['y21'] + solution['y22'] + \
                                solution['y23'] + solution['y24'] + solution['y25'])

    trace = pm.sample(draws=5000, tune=20000, chains=2, cores = 2)

_ = pm.traceplot(trace)
plt.savefig('ADM1_cali_V21/traceplot_V21.png')

summary = pm.summary(trace)
summary.to_csv('ADM1_cali_V21/summary_of_trace_ADM1_V21.csv')

with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names = ['biogas_pred', 'pH_pred', 'TAN_pred', 'tCOD_pred'], samples = 3000)

#---- PPC------
biogas_predss = posterior_predictive['biogas_pred']
DF_biogas_predss = pd.DataFrame(biogas_predss)
DF_biogas_predss.to_csv('ADM1_cali_V21/ppc_biogas.csv')

pH_predss = posterior_predictive['pH_pred']
DF_pH_predss = pd.DataFrame(pH_predss)
DF_pH_predss.to_csv('ADM1_cali_V21/ppc_pH.csv')

TAN_predss = posterior_predictive['TAN_pred']
DF_TAN_predss = pd.DataFrame(TAN_predss)
DF_TAN_predss.to_csv('ADM1_cali_V21/ppc_TAN.csv')

tCOD_predss = posterior_predictive['tCOD_pred']
DF_tCOD_predss = pd.DataFrame(tCOD_predss)
DF_tCOD_predss.to_csv('ADM1_cali_V21/ppc_tCOD.csv')
