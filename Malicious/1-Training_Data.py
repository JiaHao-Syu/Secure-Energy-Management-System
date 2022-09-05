import os
import sys
import time
import copy
import random
import numpy as np


# MS: Management System
# P / Q: Price / Quantity
# OR: Operating Reserve Rate
# SD: Supply Distribution
# FSQ/FDQ: Fail supply/demand qunatity
# Equ: Equilibrium



ID = 1     # ID of the simulation
LOBJ_SD = [   
           [0.30, 0.10, 0.10, 0.15, 0.10, 0.10, 0.15],     # S
#            [0.30, 0.05, 0.10, 0.10, 0.15, 0.15, 0.15],
#            [0.25, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15],
#            [0.25, 0.05, 0.20, 0.15, 0.10, 0.10, 0.15],     # S
#            [0.25, 0.05, 0.15, 0.10, 0.15, 0.15, 0.15],
#            [0.20, 0.10, 0.20, 0.15, 0.10, 0.10, 0.15],     # S
#            [0.20, 0.05, 0.25, 0.15, 0.10, 0.10, 0.15],
#            [0.20, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15],     # S
#            [0.15, 0.10, 0.15, 0.15, 0.15, 0.15, 0.15],
#            [0.15, 0.05, 0.10, 0.10, 0.20, 0.20, 0.20]      # S
          ]

def pri2idx(p):
    return round(p*100-1)

def idx2pri(idx):
    return (idx+1)/100.

def CurveS(Table):
    Q, Curve  = np.zeros(750, dtype=float), np.zeros(750, dtype=float)
    for p, q in Table:   Q[np.min([pri2idx(p), 750-1])] += q
    Curve[0] = Q[0]
    for i in range(1,750): Curve[i] = Curve[i-1]+Q[i]
    return Curve

def CurveD(para):
    b,h,l,r = para
    CR = np.zeros(750, dtype=float)
    for i in range(750):
        p = idx2pri(i)
        if   p<=l: CR[i] = h 
        elif p>=r: CR[i] = b 
        else:      CR[i] = b + (h-b)* (r-p)/(r-l) 
    return CR

def New_Std_Firm(T):
    rnd, typ = random.uniform(0,1), 0
    if   rnd<np.sum(Malicious_Prob[:1]): typ = -1
    elif rnd<np.sum(Malicious_Prob[:2]): typ = -2
    elif rnd<np.sum(Malicious_Prob[:3]): typ = -3
    Firm = [SCM[T], 
            SFM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SFV),
            SVM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SVV),
            SQM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SQV),
            typ, [], []]
    return Firm

# Supply-Side Initialization
def SS_Init(Sn):
    Capital, MaxQ, FixCost, VarCost, Type, HPQ, HRQ = [], [], [], [], [], [], []
    for T in range(len(Sn)):
        N = Sn[T]
        Capital += [SCM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SCV)    for i in range(N)]
        FixCost += [SFM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SFV)    for i in range(N)]
        VarCost += [SVM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SVV)    for i in range(N)]
        MaxQ    += [SQM[T]* (1+np.clip(np.random.normal(0,1), -3, 3)*SQV)    for i in range(N)]
        typ = []
        for i in range(N):
            rnd = random.uniform(0,1)
            if   rnd<np.sum(Malicious_Prob[:1]): typ.append(-1)
            elif rnd<np.sum(Malicious_Prob[:2]): typ.append(-2)
            elif rnd<np.sum(Malicious_Prob[:3]): typ.append(-3)
            else:                                typ.append( 0)
        Type    += typ
        HPQ     += [[] for i in range(N)]
        HRQ     += [[] for i in range(N)]
    return np.asarray([Capital, FixCost, VarCost, MaxQ, Type, HPQ, HRQ])

# Demand-Side Initialization
def DS_Init(Dn):
    B, H, L, R = [], [], [], []
    for T in range(len(Dn)):
        N  =   Dn[T]
        B += [ DMQ[T]*   MDR *(1+np.clip(np.random.normal(0,1), -3, 3)*DBV)  for i in range(N)]
        H += [ DMQ[T]*(1-MDR)*(1+np.clip(np.random.normal(0,1), -3, 3)*DHV)  for i in range(N)]
        L += [ DLM+              np.clip(np.random.normal(0,1), -3, 3)*DLV   for i in range(N)]
        R += [ DRM+              np.clip(np.random.normal(0,1), -3, 3)*DRV   for i in range(N)]
    return np.asarray([B, H, L, R])

def Initialization():
# Demand side parameters (Fix for all systems) 
    DP  =  DS_Init(DN)                                                           # Demand-side parameters
    FDC = []                                                                          # Fix Demand Curve
    for i in range(len(DN)):                                                          # Each type of demander
        for j in range(DN[i]):                                                        # Each demander of the type
            FDC.append(CurveD(DP[:, int(np.sum(DN[:i])+j)])) 
# Supply side parameter (Vary for all systems)
    sp  =  SS_Init(sn)                                                           # Initialized supply-side parameters
    SN  = [copy.deepcopy(sn) for idsys in range(N_Sys)]                               # Number of supplier [N_Sys, 7]
    SP  = [copy.deepcopy(sp) for idsys in range(N_Sys)]                               # Initialized supply-side parameters [N_Sys, 4, SN]
# Initialzed P & Q
    SQ_Max  = np.sum([ sp[3][i]                    for i in range(np.sum(sn))])
    SP_Init = np.sum([(sp[1][i]+sp[2][i])*sp[3][i] for i in range(np.sum(sn))]) / SQ_Max               # Average cost as initialized EquP

# Initialization for all systems
    Equ_P, LEQP   = np.zeros((N_Sys),dtype=float), np.zeros((N_Sys,N_Days),dtype=float)                # Equlibrium P / Historical Equ_P
    Equ_Q, LEQQ   = np.zeros((N_Sys),dtype=float), np.zeros((N_Sys,N_Days),dtype=float)                # Equlibrium Q / Historical Equ_Q
    Equ_QS        = np.zeros((N_Sys,len(ST)),dtype=float)                                              # Difference (Equilibrium Q from demand - from supply) / Equlibrium Q for each supply type   
    
    SUB,  DOR     = np.zeros((N_Sys,len(ST)),dtype=float), np.zeros((N_Sys),dtype=float)               # Current Sub / OR
    CSD,  COR     = np.zeros((N_Sys,len(ST)),dtype=float), np.zeros((N_Sys),dtype=float)               # Current SD / OR
    LOR,  LORD    = np.zeros((N_Sys, N_Days),dtype=float), np.zeros((N_Sys,N_Days), dtype=float)       # List of (MAE) operating reserve rate
    DASD, DHSD    = np.zeros((N_Sys, N_Days),dtype=float), np.zeros((N_Sys,N_Days), dtype=float)
    LSD,  ASD     = [[ISD] for i in range(N_Sys)], np.zeros((N_Sys,len(ST)),dtype=float)
    
    Sub_RM        = np.zeros((N_Sys),dtype=float)
    for idsys in range(N_Sys):   SUB[idsys] = np.asarray([SFM[i]+SVM[i]-SP_Init for i in range(len(ST))])
# Initialization for SYS0-QL
    tms = np.array([[1,0.5,0,0,0],[0,0.5,1,0.5,0],[0,0,0,0.5,1]])                     # Inital vale for TMS
    TMO = np.array([[1,0.5,0,0,0],[0,0.5,1,0.5,0],[0,0,0,0.5,1]])                     # Table of MS for operating reserve rate
    TMS = [copy.deepcopy(tms) for i in range(len(ST))]                                # Table of MS for subsidy
    PreSA, PreOA = [0 for i in range(len(ST))], 0                                     # Previous S/O action
    PreSS, PreOS = [0 for i in range(len(ST))], 0                                     # Previous S/O state

    return FDC, SN, SP,  Equ_P, LEQP, Equ_Q, LEQQ, Equ_QS, SUB, DOR, CSD, COR, LOR, LORD, LSD, ASD, DASD, DHSD, Sub_RM,  TMS, TMO, PreSA, PreSS, PreOA, PreOS
    
    


# Main
for idsd, OBJ_SD in enumerate(LOBJ_SD):
    OBJ_OR = 0.10     # Objective rate of operation reserve
    
    N_Sys  = 1        # Number of systems
    N_Iter = 1        # Number of iterations
    N_Days = 1825     # Number of days      100    
    TP = 50000        # Total power generation

    # Supply-Side  
    TP  = 50000                                            # Total power generation
    ST  = ['Coal','Fuel','Gas','Nuclear','Hydro','Wind','Solar']   # Supplier Type
    ISD = [0.29, 0.03, 0.36, 0.08, 0.09, 0.02, 0.13]       # Initial Supplier distribution
    SDV = [0.05, 0.05, 0.05, 0.05, 0.30, 0.30, 0.30]       # Supply variance' within dyas
    SQV, SQM = 0.2, [600, 200, 500, 1000, 100, 100,  10]   # Variance' / Mean of maximum Q     within firms
    SFV, SFM = 0.2, [  1,   1,   1,    1, 2.5, 1.4, 3.0]   # Variance' / Mean of fix cost      within firms
    SVV, SVM = 0.2, [0.5, 4.0, 0.8,  0.3, 0.0, 0.0, 0.0]   # Variance' / Mean of variable cost within firms
    SCV, SCM = 0.2, [100*sqm for sqm in SQM]               # Variance' / Mean of capital       within firms
    sn = [round(TP * ISD[i]/SQM[i]) for i in range(len(ISD))] # Number of supplier
    Malicious_Prob = [0.1, 0.1, 0.1]

    # Demand-Side
    DT  = ['Household', 'Commercial', 'Industrial']        # Demander Type
    MDR =  0.25                                            # Minimum demand ratio
    DD  = [0.20, 0.15, 0.65]                               # Demander distribution
    DDV = [0.10, 0.10, 0.20]                               # Demand variance' within days
    DMQ = [  20,  100, 2000]                               # Maximum of demand Q
    DBV, DHV = 0.2, 0.2                                    # Variance' of B and H  within firms
    DLM, DLV = 0, 0.5                                      # Mean / Varicance of R within firms
    DRM, DRV = 5, 0.5                                      # Mean / Varicance of O within firms
    DN = [round(TP*(1-OBJ_OR) *DD[i] /DMQ[i] ) for i in range(len(DD))]  # Number of demander

    # Management System
    QSA, QOA = [0.4, 0.2, 0, -0.1, -0.2], [0.04, 0.02, 0, -0.01, -0.02]    # QL Subsidy / OR Action


    RanDay, RanPro = 30, 0.05
    alpha          = 0.75
    alphaQ, gammaQ = 0.5, 0.2
    Converge  = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Converge days
    Fail_Rate = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Fail rate
    Mean_EQP  = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Mean     of Equ. price
    Mean_EQQ  = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Mean     of Equ. Q
    Mean_ORR  = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Mean     of operating reserve rate
    Var_EQP   = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Variance of Equ. price
    Var_EQQ   = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Variance of Equ. Q
    Var_ORR   = np.zeros((N_Sys, N_Iter), dtype=float)                                   # Variance of operating reserve rate
    MAE_ORR   = np.zeros((N_Sys, N_Iter), dtype=float)                                   # MAE      of operating reserve rate
    MAE_DSD   = np.zeros((N_Sys, N_Iter), dtype=float)                                   # MAE      of difference of supply distribution

    for idi in range(N_Iter):
        # Initialization
        FDC, SN, SP,  Equ_P, LEQP, Equ_Q, LEQQ, Equ_QS, SUB, DOR, CSD, COR, LOR, LORD, LSD, ASD, DASD, DHSD, Sub_RM,  TMS, TMO, PreSA, PreSS, PreOA, PreOS = Initialization()

        # Simulation    
        for idd in range(N_Days):                           # For each day
        # Supply-Side
            Curve_S, Ask_Table = np.zeros((N_Sys,750), dtype=float), [[] for idsys in range(N_Sys)]
            FSQ, FSQL          = np.zeros((N_Sys    ), dtype=float), [[] for idsys in range(N_Sys)]
            for idsys in range(N_Sys):
                for idst in range(len(ST)):                 # Each type of supplier
                    for idf in range(SN[idsys][idst]):      # Each firm of the type
                        ids = int(np.sum(SN[idsys][:idst])+idf)
                        ca, fc, vc, mq, tp, hpq, hrq = SP[idsys][:, ids]
                        if ca<0:                            # Bankruptcy
                            Ask_Table[idsys].append([7.5, 0])
                            FSQL[idsys].append(0)
                            continue
                        PQ = mq  if (Equ_P[idsys]>vc-SUB[idsys][idst] or Equ_P[idsys]==0)  else 0        # Plan Q (Production Strategy !!!)
                        fs = np.min([abs(np.random.normal(0, SDV[idst])), 1])                            # Failed Production
                        if   tp==-1: fs=1.0
                        elif tp==-2: fs=0.5
                        elif tp==-3: 
                            rdn = random.uniform(0,1)
                            if   rdn<1/3.: fs=0
                            elif rdn<2/3.: fs=0.5
                            else:          fs=1.0
                        FSQ[idsys] += PQ*fs
                        FSQL[idsys].append(PQ*fs)
                        cost        = fc*mq + (vc-SUB[idsys][idst])*PQ
                        if PQ!=0: Ask_Table[idsys].append([round(np.max([cost/PQ, 0.01]), 2), PQ])
                        else:     Ask_Table[idsys].append([7.5, 0])
                        SP[idsys][-1][ids] = SP[idsys][-1][ids]+[PQ*(1-fs)]
                        SP[idsys][-2][ids] = SP[idsys][-2][ids]+[PQ]
                Curve_S[idsys] = CurveS(Ask_Table[idsys])

        # Demand-Side
            Curve_D, FDQ = np.zeros((N_Sys,750), dtype=float), np.zeros((750), dtype=float)
            for i in range(len(DN)):                        # Each type of demander
                for j in range(DN[i]):                      # Each demander of the type
                    CD = copy.deepcopy(FDC[ int(np.sum(DN[:i])+j) ])
                    sd = np.clip(np.random.normal(0, DDV[i]), -1, 1)
                    FDQ += CD*sd
                    for idsys in range(N_Sys):  Curve_D[idsys] += CD*(1+DOR[idsys])

        # Market Equilibrium
            for idsys in range(N_Sys):
                idx = 749
                for i in range(749):
                    if Curve_S[idsys][i+1] > Curve_D[idsys][i+1]:
                        idx = i
                        break
                Equ_P[idsys], Equ_Q[idsys] = idx2pri(idx), Curve_D[idsys][idx]
                LEQP[idsys][idd], LEQQ[idsys][idd] = Equ_P[idsys], Equ_Q[idsys]



        # Update Captial of Firms & Subsidy Residual & Operation Reserve
            for idsys in range(N_Sys):                          # Each system
                idx = pri2idx(Equ_P[idsys])
                Sub_RM[idsys] += Equ_P[idsys]*( (Curve_D[idsys][idx]+FDQ[idx]*(1+DOR[idsys])) - Curve_S[idsys][idx] )

                for idst in range(len(ST)):                     # Each type of supplier
                    eq, tq = 0, 0                               # Equilbrium Q, total Q
                    for idf in range(SN[idsys][idst]):          # Each firm of the type
                        ids = int(np.sum(SN[idsys][:idst])+idf) # Index of the firm
                        if SP[idsys][0][ids]>0:                 # Alive firm
                            p, q = Ask_Table[idsys][ids]
                            q  -= FSQL[idsys][ids]
                            tq += q
                            if p<=Equ_P[idsys]: eq += q         # If deal
                            profit = (Equ_P[idsys]-p)*q  if p<=Equ_P[idsys]  else -(p+SUB[idsys][idst])*q    # !!! No subsidy for whom is not deal
                            SP[idsys][0][ids] += profit         # Update capital
                            Sub_RM[idsys] += FSQL[idsys][ids]*p
                            # print(p<=Equ_P[idsys], profit, (Equ_P[idsys]-p), p, -(p+SUB[idsys][idst]), q)
                    Equ_QS[idsys][idst] = eq
                    Sub_RM[idsys] -= SUB[idsys][idst]*eq
                COR[ idsys] = 1 - ( (Equ_Q[idsys]/(1+DOR[idsys])+FDQ[idx]) / (Curve_S[idsys][-1]-FSQ[idsys]) )          # Current OR     
                LOR[ idsys][idd] = COR[idsys]
                LORD[idsys][idd] = abs(OBJ_OR - COR[idsys])



        # Supplied Distribution
            for idsys in range(N_Sys):
                SumSQ =  np.sum(Equ_QS[idsys])
                CSD[idsys] = [Equ_QS[idsys][i]/SumSQ  for i in range(len(ST))]                               # Current SD
                if np.isnan(CSD[idsys]).any():
                    print(Equ_QS[0])
                    print(SumSQ, Equ_P[0], Equ_Q[0])
                    print(Curve_S[0])
                    print(DOR[0], SUB[0])
                    print(ID, '\n\n')
                    sys.exit()
                LSD[idsys].append(copy.deepcopy(CSD[idsys]))
                LSD[idsys] = LSD[idsys][-30:]
                ASD[idsys] = np.mean(LSD[idsys], axis=0)
                DASD[idsys][idd] = np.sum([abs(OBJ_SD[i]-ASD[idsys][i]) for i in range(len(ST))])
                DHSD[idsys][idd] = np.sum([abs(OBJ_SD[i]-CSD[idsys][i]) for i in range(len(ST))])

        # Update Subsidy & OR: QL
            # Rewards
            rewardo, rewards = 0, []
            if (LOR[0][idd-1]-OBJ_OR)*(LOR[0][idd-1]-LOR[0][idd])<0:   rewardo = 1 + (LOR[0][idd-1]-OBJ_OR)*(LOR[0][idd-1]-LOR[0][idd])
            else:                                                      rewardo = 1 + np.max([1-abs((LOR[0][idd-1]-LOR[0][idd])/(LOR[0][idd-1]-OBJ_OR)) , 0])
            for idst in range(len(ST)):
                if (LSD[0][-2][idst]-OBJ_SD[idst])*(LSD[0][-2][idst]-LSD[0][-1][idst])<0:   rewards.append(1 + (LSD[0][-2][idst]-OBJ_SD[idst])*(LSD[0][-2][idst]-LSD[0][-1][idst]))
                else:                                                                       rewards.append(1 + np.max([1-abs((LSD[0][-2][idst]-LSD[0][-1][idst])/(LSD[0][-2][idst]-OBJ_SD[idst])) , 0]))

            # State of subsidy
            for idst in range(len(ST)):          # Each type of supplier
                CurSS = 1
                if CSD[0][idst] < OBJ_SD[idst]*0.8: CurSS=0
                if CSD[0][idst] > OBJ_SD[idst]*1.2: CurSS=2
                TMS[idst][PreSS[idst]][PreSA[idst]] = (1-alphaQ)*TMS[idst][PreSS[idst]][PreSA[idst]] + alphaQ*( rewards[idst] + gammaQ*np.max(TMS[idst][CurSS]) )
                if np.random.random()<=RanPro and idd<RanDay:  CurSA = np.random.randint(len(QSA))
                else:   CurSA = np.argmax(TMS[idst][CurSS])
                SUB[0][idst] = (1-alpha)*SUB[0][idst] + alpha*(SUB[0][idst]+QSA[CurSA])
                SUB[0][idst] = np.clip(SUB[0][idst], -Equ_P[0]+SVM[idst], SFM[idst]+SVM[idst])
                PreSA[idst], PreSS[idst] = CurSA, CurSS
            EstGain, EstCost = 0, 0
            for idst in range(len(ST)):          # Each type of supplier
                if SUB[0][idst]<0:   EstGain -= SUB[0][idst] * Equ_QS[0][idst]
                else:                EstCost += SUB[0][idst] * Equ_QS[0][idst]
            if Sub_RM[0]+EstGain<EstCost:
                factor = max([(Sub_RM[0]+EstGain)/EstCost, 0])
                for idst in range(len(ST)):      # Each type of supplier
                    if SUB[0][idst]>0:   SUB[0][idst]*=factor

            # State of OR
            CurOS = 1
            if COR[0] < OBJ_OR*0.9: CurOS=0
            if COR[0] > OBJ_OR*1.1: CurOS=2
            TMO[PreOS][PreOA] = (1-alphaQ)*TMO[PreOS][PreOA] + alphaQ*( rewardo + gammaQ*np.max(TMO[CurOS]) )
            if np.random.random()<=RanPro and idd<RanDay:  CurOA = np.random.randint(len(QOA))
            else:   CurOA = np.argmax(TMO[CurOS])
            DOR[0] = np.clip((1-alpha)*DOR[0] + alpha*(DOR[0]+QOA[CurOA]), 0, 1)
            PreOA, PreOS = CurOA, CurOS
            # print("{:d} {:.2f} {:d} {:.2f} ~ {:.2f} {:.2f}".format(idd, Equ_P[0], int((Equ_Q[0])/1000), DOR[0], COR[0], DASD[idsys][idd]))


            # New suppliers
            # if idd%30==29:
            for idsys in range(N_Sys):
            # New suppliers by split
                SumSQ, NQ =  np.sum(Equ_QS[idsys]), np.zeros(len(ST), dtype=float)
                for idst in range(len(ST)):                          # Each type of supplier
                    if CSD[idsys][idst]>OBJ_SD[idst]: continue       # Don't split if current distribution > objective.   
                    nq = SumSQ * (OBJ_SD[idst]-CSD[idsys][idst])     # New Q quota
                    for idf in range(SN[idsys][idst]):               # Each firm of the type
                        if nq<=0: break
                        ids = int(np.sum(SN[idsys][:idst])+idf)
                        if SP[idsys][0][ids] >= 12*SCM[idst]:    # Rich enough to split
                            # print('New', ST[idst], 'firm by split!', SP[idsys][0][ids], 12*SCM[idst])
                            invec     = New_Std_Firm(idst)       # Insert vector
                            SP[idsys] = np.insert(SP[idsys], int(np.sum(SN[idsys][:idst+1])), [invec], axis=1)
                            SN[idsys][idst] += 1
                            nq                -= invec[-4]
                            SP[idsys][0][ids] -= 11*SCM[idst]
                    NQ[idst] = SumSQ *(OBJ_SD[idst]-CSD[idsys][idst]) - nq

            # New suppliers by MS
                if Sub_RM[idsys] > 0:
                    Difp = [ CSD[idsys][i]/OBJ_SD[i]  for i in range(len(ST))]
                    i   = np.argsort(Difp)[0]
                    if Difp[i]<1 and Sub_RM[idsys] >= 11*SCM[i]:
                        # print('New', ST[i], 'firm by MS!')
                        invec          = New_Std_Firm(i)         # Insert vector
                        SP[idsys]      = np.insert(SP[idsys], int(np.sum(SN[idsys][:i+1])), [invec], axis=1)
                        SN[idsys][i]  += 1
                        Sub_RM[idsys] -= 11*SCM[i]

                if DASD[idsys][idd]<0.1 and Converge[idsys][idi]==0:   
                    Converge[idsys][idi] = idd
                    # print('Converge !!!')
                if DASD[idsys][idd]>0.1 and Converge[idsys][idi]!=0:   
                    Converge[idsys][idi] = 0
                    # print('Broke !!!')


        for idsys in range(N_Sys):
            if Converge[idsys][idi]==0:   Converge[idsys][idi] = N_Days
            Fail_Rate[idsys][idi] = np.sum([1 for i in LOR[idsys] if i<0])/len(LOR[idsys])
            Mean_EQP[ idsys][idi] = np.mean(LEQP[idsys])
            Mean_EQQ[ idsys][idi] = np.mean(LEQQ[idsys])
            Mean_ORR[ idsys][idi] = np.mean(LOR[ idsys])
            MAE_ORR[  idsys][idi] = np.mean(LORD[idsys])
            MAE_DSD[  idsys][idi] = np.mean(DHSD[idsys])
            Var_EQP[  idsys][idi] = np.std(LEQP[ idsys])
            Var_EQQ[  idsys][idi] = np.std(LEQQ[ idsys])
            Var_ORR[  idsys][idi] = np.std(LOR[  idsys])
            
    np.save('Training_Dataset/HPQ-'+str(int(OBJ_OR*100))+'-'+str(idsd)+'-'+str(ID)+'.npy', SP[0][-2])
    np.save('Training_Dataset/HRQ-'+str(int(OBJ_OR*100))+'-'+str(idsd)+'-'+str(ID)+'.npy', SP[0][-1])