# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %%writefile C:\Users\Cflhxb\AppData\Local\anaconda3\Lib\CAMP.py
    
import datetime as dt
import pandas as pd
import numpy as np
import math as math 
import MathsUtilities as MUte
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import curve_fit

CampConstants = {
    'k':-0.17,
    'VIThreshold':1.0,
    'TSThreshold':2.0,
    'SlopeFLNvsTS':1.1,
    'IntFLNvsTS':2.85,
    'PpMax':16,
    'PpMin':8,
}

colors = ['r','b','b','r']
fills = ['w','b','w','r']

def CalcdHS(Tt,Pp,HS,BasePhyllo,PhylloPpSens):
    """Calculate daily increase in Haun stage
    Args:
        Tt: Thermal time increment
        Pp: The current photoperiod
        HS: Current Haun Stage
        BasePhyllo: Base phyllochron at 16h Pp between 3 and 7 HS
        PhylloPpSens: Photo period sensitivity of phyllochron.  Relative increase in phyllochron as 8h Pp
    Returns:
        Daily increas in Haun Stage
    """
    HS = max(0,HS)
    StageFactor = np.interp(HS,[0,2,2.0001,7,7.0001,12],[0.75,0.75,1.0,1.0,1.4,1.4],False)
    PpFactor = 1 + PhylloPpSens * np.interp(Pp,[0,8,12,20],[1,1,0,0],False)
    Phyllochron = BasePhyllo*StageFactor*PpFactor
    return Tt/Phyllochron

def CalcPpResponse(Pp, Base, Max):
    """ Calculate value of a photoperiod sensitive variable between Max and Baee values
    Args:
        Pp: Photoperiod
        Base: value below 8h Pp
        Max: value above 16h Pp
        dPB: delta base phyllochrons
    Returns:
        variable value dependent on photoperiod and adjusted for dHS
    """
    if Pp <= CampConstants['PpMin']:
        return Base
    if (Pp > CampConstants['PpMin']) and (Pp < CampConstants['PpMax']):
        return (Base + (Max-Base) * (Pp-CampConstants['PpMin'])/(CampConstants['PpMax']-CampConstants['PpMin']))
    if (Pp >= CampConstants['PpMax']):
        return Max

def CalcBaseUpRegVrn1(Tt, dHS, BaseDVrn1):
    """ Calculate upregulation of base Vrn1
    
    Args:
        Tt: Thermal time increment
        dHS: Delta HaunStage
        BaseDVrn1: coeffociennt for Base Vrn1 expression
        
    Returns:
        delta BaseVrn1 representing the additional Vrn1 expression from base expression
    """
    if Tt < 0: 
        BaseDVrn1 = 0
    return BaseDVrn1 * dHS

def CalcColdUpRegVrn1(Tt,dHS, MaxVrn1, k):
    """ Upregulation of Vrn1 from cold.  Is additional to base vrn1
        BaseDVrn1 in seperate calculation otherwise te same as Brown etal 2013
    Args:
        Tt: Thermal time increment
        dHS: Delta Base Phyllochrons
        MaxVrn1: coefficient for Maximum upregulation of Vrn1
        k: The exponential coefficient determining temperature response
        
    Returns:
        delta ColdVrn1 representing the additional Vrn1 expression from cold upregulation
    """
    UdVrn1 = MaxVrn1 * np.exp(k*Tt)
    if (Tt < 20):
        return UdVrn1 * dHS
    else:
        return -5
        
def calcTSHS(FLN,IntFLNvsTSHS,SlopeFLNvsTS):
    """Haun stage timing of terminal spikelet 
       Inverts equation 5 from Brown etal 2013 FLN =  2.85 + 1.1*TSHS and converts it to base phyllochrons.
       Note the intercept differs, was typeo on publication
       Intercept has been made variable for as needs to be lower for some very fast varieties
    Args:
        FLN: The final leaf number
        IntFLNvsTSHS: The intercept of the regression between FLN and TSHS
    Returns:
        Estimation of number of base phyllochrons to terminal spikelet
    """
    return (FLN - IntFLNvsTSHS)/SlopeFLNvsTS

def CAMPmodel(Out, Day, Tt, Pp, Params, Consts, TtEmerge):
    """ The Cereal Anthesis Molecular Phenology model.
        Based on the ideas presented in Brown etal 2014 (Annals of Botany)
        Alterations made replacing Vrn4 notion with methalation of Vrn1, allowing Vrn3 to act before VI and changint the pattern for Vrn2 expression
        Other alterations to implement working code base
    Args:
        Out: "FLN" or None.  If "FLN" will only return estimated FLN else will return full dataframe with daily state variable values
        Day: List, 1:EndDay representing the timesteps in model run
        Tt: List of same length as Day, representing daily temperature
        Pp: List of same length as Day, representing daily photoperiod
        Params: Dict with genotype parameters fitted by CalcCultivarVrnCoeffs() 
        Consts: Dict with crop specific constants,
        TtEmerge: Thermaltime from sowing to emergence
    """
    # Set up Data structure and initialise values
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    DF = pd.DataFrame(index = [0], columns = ['Day','Tt','Pp'])
    IsGerminated = False
    IsEmerged = False
    IsMethalating = False
    IsVernalised = False
    IsReproductive = False
    IsAtFlagLeaf = False
    DF.loc[0,'Day'] = 0
    DF.loc[0,'Tt'] = Tt[0]
    DF.loc[0,'RelCold'] = 0
    DF.loc[0,'Pp'] = Pp[0]
    DF.loc[0,'fPP'] = 0
    DF.loc[0,'Stage'] = 'Germination'
    DF.loc[0,'IsGerminated'] =True
    DF.loc[0,'IsEmerged'] = False
    DF.loc[0,'IsVernalised'] = False
    DF.loc[0,'IsReproductive'] = False
    DF.loc[0,'IsAtFlagLeaf'] = False
    DF.loc[0,'HS'] = -TtEmerge/Params.BasePhyllo
    DF.loc[0,'dHS'] = 0
    DF.loc[0,'AccumTt'] = Tt[0]
    DF.loc[0,'VrnB'] = 0
    DF.loc[0,'dVrnB'] = 0
    DF.loc[0,'MaxVrn'] = 0
    DF.loc[0,'dMaxVrn'] = 0
    DF.loc[0,'Cold'] = 0
    DF.loc[0,'dCold'] = 0
    DF.loc[0,'ApDev'] = 0
    DF.loc[0,'MaxVrn2'] = 0
    DF.loc[0,'Vrn2'] = 0.0
    DF.loc[0,'AccumSPp'] = 0.0
    #DF.loc[0,'Vrn2ef'] = 0.0
    DF.loc[0,'Vrn1'] = 0.0
    DF.loc[0,'dVrn1'] = 0
    DF.loc[0,'Vrn3'] = 0.0
    DF.loc[0,'dVrn3'] = 0
    DF.loc[0,'VIHS'] = 0
    DF.loc[0,'TSHS'] = 0
    DF.loc[0,'FLN'] = Consts['IntFLNvsTS']
    
    d = 1
    
    # Model daily loop
    # ^^^^^^^^^^^^^^^^
    while (IsAtFlagLeaf == False) and (d < Day[-1]):
        ColdYesterday = 0
        ColdYesterday = DF.loc[d-1,'Cold']
        DF.loc[d,:] = DF.loc[d-1,:] #Carry yesterdays values over to today
        DF.loc[d,'Stage'] = np.nan
            
        DF.loc[d,'Day'] = d
        # Set daily environment variables
        DF.loc[d,'Tt'] = Tt[d]
        DF.loc[d,'Pp'] = Pp[d]
        #Zero set deltas
        DF.loc[d,'dHS'] = 0
        DF.loc[d,'dVrnB'] = 0
        DF.loc[d,'dMaxVrn'] = 0
        DF.loc[d,'dCold'] = 0
        DF.loc[d,'Vrn2'] = 0
        DF.loc[d,'dVrn1'] = 0
        DF.loc[d,'dVrn3'] = 0
        
                
        DF.loc[d,'AccumTt'] = DF.loc[d,'AccumTt'] + DF.loc[d,'Tt']
        
        PropnOfDay = 1.0
        if (DF.loc[d,'AccumTt'] > TtEmerge) and (IsEmerged==False):
            IsEmerged = True
            DF.loc[d,'Stage'] = 'Emergence'
            PropnOfDay = (DF.loc[d,'AccumTt'] - TtEmerge)/DF.loc[d,'Tt'] # Calculate fraction of emergence days Tt that is not used for emergence
        
        # Calculate daily Haun Stage changes
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if IsEmerged==False: # Crop not yet emerged 
            EmergDurationFactor = 1
            if (DF.loc[d,'AccumTt'] > 90):  # Calculate EmergenceDurationFactor to slow accumulation of HS if emergence is taking a long time.  This slows Vrn1 expression under slow emergence and strange responses to delayed sowing
                EmergDurationFactor = np.exp(-0.015 * (DF.loc[d,'AccumTt']-90))
            DF.loc[d,'dHS'] = DF.loc[d,'Tt']/Params.BasePhyllo  * EmergDurationFactor
            DF.loc[d,'fPP'] = 0
            DF.loc[d,'Vrn2'] = 0
            
        else: # Crop emerged
            # Calculate delta haun stage
            DF.loc[d,'dHS'] = CalcdHS(DF.loc[d,'Tt'],DF.loc[d,'Pp'],DF.loc[d,'HS'],Params.BasePhyllo,Params.PhylloPpSens)
            
        #increment HS
        DF.loc[d,'HS'] += DF.loc[d,'dHS']
        
        # Calculate Vrn gene expression
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if (IsReproductive==False):
            if (IsVernalised==False): #set as rates and factors for vegetative phase
                DF.loc[d,'dVrnB'] = Params.rVrnBVeg * DF.loc[d,'dHS']
                DF.loc[d,'dMaxVrn'] = Params.rVrnMVeg * DF.loc[d,'dHS']
                DF.loc[d,'PpVrn3Fact'] = Params.rVrn3Veg
            else:
                 #set as rates and factors for early reproductive phase
                DF.loc[d,'dVrnB'] = Params.rVrnBER * DF.loc[d,'dHS']
                DF.loc[d,'dMaxVrn'] = Params.rVrnMER * DF.loc[d,'dHS']
                DF.loc[d,'PpVrn3Fact'] = Params.rVrn3ER
            #Increment BaseVern and MaxVern
            DF.loc[d,'VrnB'] += DF.loc[d,'dVrnB']
            DF.loc[d,'MaxVrn'] = min(2,DF.loc[d,'MaxVrn'] + DF.loc[d,'dMaxVrn'])

            # Calculate daily cold response
            if (IsVernalised==False):
                DF.loc[d,'RelCold'] = CalcColdUpRegVrn1(DF.loc[d,'Tt'],1,1,Consts['k'])
                DF.loc[d,'dCold'] = DF.loc[d,'RelCold'] * DF.loc[d,'dVrnB'] * Params.rVrn1
                DF.loc[d,'Cold'] = max(0.0,DF.loc[d,'Cold'] + DF.loc[d,'dCold'])
                if DF.loc[d,'Cold'] > Params.MethalationThreshold:
                    DF.loc[d,'dVrn1'] = max(0,(DF.loc[d,'Cold'] - Params.MethalationThreshold)-(ColdYesterday - Params.MethalationThreshold))
            # Increment Vrn 1
            DF.loc[d,'Vrn1'] += DF.loc[d,'dVrn1']

            # Calcualte expression of photoperiod sensitive genes
            if (IsEmerged==True):  # Photoperiod sensitive genes only express after emergence
                DF.loc[d,'fPP'] = CalcPpResponse(DF.loc[d,'Pp'],0,1) #relative Pp, scaled between 0 at lower threshold and 1 at upper threshold
                DF.loc[d,'AccumSPp'] += DF.loc[d,'dVrnB'] * (1-DF.loc[d,'fPP'])  #accumulate short day exposure for reducing mVrn2
                DF.loc[d,'MaxVrn2'] = max(0,(Params.mVrn2 * DF.loc[d,'fPP']) - DF.loc[d,'AccumSPp'])
                DF.loc[d,'Vrn2'] = max(0,DF.loc[d,'MaxVrn2'] - (DF.loc[d,'VrnB'] + DF.loc[d,'Vrn1'])) 
                if (DF.loc[d,'Vrn2'] == 0): # express vrn3 relative to Pp if effective Vrn2 is down regulated to zero
                    DF.loc[d,'dVrn3'] = (DF.loc[d,'PpVrn3Fact']-1) * DF.loc[d,'fPP'] * DF.loc[d,'dVrnB']
                #Increment Vrn3
            DF.loc[d,'Vrn3'] += DF.loc[d,'dVrn3']

            # Increment todays Vrn expression values using deltas just calculated
            if (IsVernalised==False): #If not vernalised need to include Vrn2 and constrain to maxrate
                DF.loc[d,'ApDev'] = min(DF.loc[d,'MaxVrn'],max(DF.loc[d,'VrnB'],DF.loc[d,'VrnB'] + DF.loc[d,'Vrn1'] + DF.loc[d,'Vrn3'] - DF.loc[d,'MaxVrn2'])) 
            else:
                DF.loc[d,'ApDev'] = DF.loc[d,'ApDev'] + DF.loc[d,'dVrnB'] + DF.loc[d,'dVrn3']
        else:
            # continue accumulating VrnB after reproductive for graphing purposes
            DF.loc[d,'dVrnB'] = Params.rVrnBER * DF.loc[d,'dHS']
            DF.loc[d,'VrnB'] = min(2,DF.loc[d,'dVrnB']+DF.loc[d,'VrnB'])
        # Set Haun stage variables
        # ^^^^^^^^^^^^^^^^^^^^^^^^
        if IsVernalised == False:
            DF.loc[d,'VIHS'] = DF.loc[d,'HS']
        if IsReproductive == False:
            DF.loc[d,'TSHS'] = DF.loc[d,'HS']
            DF.loc[d,'FLN'] = Consts['IntFLNvsTS'] + Consts['SlopeFLNvsTS'] * DF.loc[d,'TSHS']


                    
        # Finally determine phenological stage
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Vernalisation saturation occurs when Vrn1 > the vernalisation threshold and Vrn2 expression is zero
        if (IsEmerged==True) and (DF.loc[d,'ApDev'] >= Consts['VIThreshold']) and (IsVernalised==False):
            IsVernalised  = True
            DF.loc[d,'Stage'] = 'Vern Sat'
            #DF.loc[d,'Vrn1'] = 0
        
        if (DF.loc[d,'ApDev'] >=  Consts['TSThreshold']) and (IsReproductive == False):
            IsReproductive = True;
            DF.loc[d,'Stage'] = 'Term Spike'
        
        #Work out if Flag leaf has appeared.
        if DF.loc[d,'HS'] >= DF.loc[d,'FLN']:
            IsAtFlagLeaf = True    
            DF.loc[d,'Stage'] = 'FlagLeaf'
    
        #Add states to dataframe
        IsDayOfEmergence = False
        
        DF.loc[d,'IsGerminated'] = IsGerminated
        DF.loc[d,'IsEmerged'] = IsEmerged
        DF.loc[d,'IsVernalised'] = IsVernalised
        DF.loc[d,'IsReproductive'] = IsReproductive
        DF.loc[d,'IsAtFlagLeaf'] = IsAtFlagLeaf
    
        # Increment day
        d += 1
        
    if (Out == 'FLN') and (IsAtFlagLeaf == True):
        return DF.iloc[-1,:]['FLN']
    else:
        return DF
    
def deriveVrnParams(c):
    data = pd.Series(dtype=np.float64)
    # Calculate the accumulated base phyllochrons at terminal spikelet for each treatment
    for pv in ['WS','CL','CS','WL']:
        data['TS_'+pv] = calcTSHS(c['FLN_'+pv],CampConstants['IntFLNvsTS'],CampConstants['SlopeFLNvsTS'])
    # Base Phyllochron
    data['BasePhyllo'] = c['BasePhyllo']
    # Phyllochron photoperiod sensitivity.  #This is not used in calculations but added so can be checked in outputs
    data['PhylloPpSens'] = c['PhylloPpSens']
    # Base Phyllochron duration of the Emergence Phase
    data['EmergDurat'] = c['TtEmerge']/data['BasePhyllo']
    # Base Phyllochron duration of vernalisation treatment
    data['VernTreatDurat'] = (c['VrnTreatDuration']*c['VrnTreatTemp'])/data['BasePhyllo']
    # Accumulated Base Phyllochrons when vernalisation treatment ended 
    data['EndVernTreat'] =  -data['EmergDurat'] + data['VernTreatDurat']
    # The minimum accumulated haunstage at which Vern saturation can occur
    data['MinVI'] = 1.1
    # The minimum haun stage duration from vern saturation to terminal spikelet
    data['MinVI->TS'] = min(3,data['TS_CL']-data['MinVI'])  #From Lincoln controlled environment data
    
    # Calculate the accumulated base phyllochrons at vernalisation saturation for each treatment
    data['VI_CL'] = data['TS_CL'] - data['MinVI->TS']   
    data['VI_WL'] = max(data['TS_WL'] - data['MinVI->TS'],data['VI_CL']) #Constrained so not earlier than the CL treatment
    data['VI_CS'] = min(data['VI_CL'],data['TS_CS']-data['MinVI->TS']) # Assume happens at the same time as VI_CL but not sooner than TS and MinVI->TS would allow 
    data['VI->TS_CS'] = data['TS_CS'] - data['VI_CS']
    data['VI_WS'] = max(data['TS_WS'] - data['VI->TS_CS'],data['VI_CL']) #Constrained so not earlier than the CL treatment
    
    # Calculate base and maximum rates and Pp sensitivities
    # Base Vrn delta during vegetative phase.  Assuming base Vrn expression starts at sowing and reaches 1 at VI for the WS treatment where no cold or Vrn3 (short Pp) to upregulate 
    data['rVrnBVeg'] = 1/(data['VI_WS']+data['EmergDurat'])
    # The fastest rate that Vrn can accumulate to reach saturation (VI).
    data['rVrnMVeg'] = 1/(data['VI_CS']+data['EmergDurat'])
    # Base Vrn delta during early reproductive phase.  Assuming Vrn expression increases by 1 between VI and TS and proceeds at a base rate where not Vrn3 upregulation (short PP)
    data['rVrnBER'] = 1/(data['TS_WS']-data['VI_WS'])
    # The maximum Vrn delta during the early reproductive phase.  Assuming Vrn increases by 1 bewtween VI and TS and proceeds at the maximum rate under long photoperiod
    data['rVrnMER'] = 1/data['MinVI->TS']
    # The relative increase in delta Vrn cuased by Vrn3 expression under long photoperiod during early reproductive phase
    data['rVrn3ER'] = data['rVrnMER']/data['rVrnBER']
    # The relative increase in delta Vrn1 caused by Vrn3 expression under long Pp during vegetative phase.  Same as rVrn3ER unless VI_WL is small
    data['rVrn3Veg'] = (1/data['VI_WL'])/data['rVrnBVeg'] 
    data['rVrn3Veg'] = max(data['rVrn3ER'],data['rVrn3Veg']) 
    
    # Calculate Maximum Vrn2 expression under long photoperiods
    # The rate of Vrn expression before VI under long Pp when Vrn2 = 0 is the product of Vrn3 and base Vrn
    # Therefore the haun stage duration when effective baseVrn x Vrn3 is up regulating during the vegetative phase under long photoperiod without vernalisation is a funciton of the VI target (1) and the rate
    data['URVrn3HSWL'] = 1/(data['rVrnBVeg'] * data['rVrn3Veg'])
    # The accumulated Haun stages when Vrn2 expression is ended and effective baseVrn x Vrn3 expression starts under long Pp without vernalisation is vrn1xVrn3Durat prior to VI
    data['DRVrn2HSWL'] = max(0,data['VI_WL'] - data['URVrn3HSWL']) 
    # The Vrn2 Expression that must be matched by Vrn1 before effective Vrn3 expression starts.  
    data['mVrn2'] = (data['DRVrn2HSWL']+data['EmergDurat']) * data['rVrnBVeg']
    
    # Calculate cold upregulation of Vrn1
    # The amount of protaganistic vrn expression required to reach VI
    data['VrnPVI_CL'] = CampConstants['VIThreshold'] + data['mVrn2']
    # The amount of Vrn expression at base rate in the CL treatment between sowing and when Vrn2 is suppressed 
    data['VrnBVeg_CL'] = (data['EmergDurat']+data['VI_CL']) * data['rVrnBVeg']
    # The amount of persistant (methalated) Vrn1 upregulated due to cold at the end of the vernalisation treatment for CL
    # Is VrnPVI_CL less base vrn expression up to VI.  Assuming Vrn3 expression prior to VI is neglegable in this case 
    data['pVrn1_CL'] = data['VrnPVI_CL'] - data['VrnBVeg_CL'] 
    # Cold threshold required before Vrn1 starts methalating.  Based on Brooking and Jamieson data the lag duration is the same length as the duration of response up to full vernalisation
    # Therefore we assume the methalation threshold must be the same size as the amount of vrn1 that is required to give full persistend vernalisation response.
    data['MethalationThreshold'] = data['pVrn1_CL']
    # Haun duration of Vrn1 upregulation, may be less than treatment duration if treatment goes past VI
    data['URVrn1HS_CL'] =  min(data['VernTreatDurat'],data['VI_CL']+data['EmergDurat'])
    # The rate of Vrn1 expression under cold treatment calculated from the amount of cold vrn1 up regulation apparante plus the methalation threshold
    data['DVrn1_CL'] = (data['pVrn1_CL']+data['MethalationThreshold'])/data['URVrn1HS_CL'] 
    # The upregulation of Vrn1 expression above base rate at 0oC
    data['DVrn1Max'] = data['DVrn1_CL']/ np.exp(c['k'] * c['VrnTreatTemp']) 
    # The relative increase in delta Vrn1 caused by cold upregulation of Vrn1
    data['rVrn1'] = data['DVrn1Max'] /data['rVrnBVeg']
                                 
    return data

def plotFLNs(CL,CS,WS,WL,Axis,fs,maxFLN=20, ylab=True):
    width = 0.4
    ind = np.arange(4) + width
    plt.bar(ind+.4, [CL,CS,WS,WL], width,
            edgecolor=['b','b','r','r'], color = ['b','w','w','r'],
            linewidth=3,alpha=0.3)
    if ylab == True:
        plt.ylabel('Final Leaf Number',fontsize=12)
    plt.tick_params(labelsize=fs)
    Axis.set_xticks(ind+width)
    Axis.set_xticklabels(['$CL$','$CS$', '$WS$', '$WL$'])
    plt.ylim(0,maxFLN)
    plt.xlim(0.5,4.1)
    Axis.spines['right'].set_visible(False)
    Axis.spines['top'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=True,top=False, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True,right=False, labelleft=True)

def addArrow(Xs,Ys,col='k'):
    bx = Xs[0]
    by = Ys[0]
    dx = Xs[1] 
    dy = Ys[1]
    prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.5",color=col,lw=1,ls=':',
            shrinkA=0,shrinkB=0)
    plt.annotate("",xy=(dx,dy),xytext=(bx,by), arrowprops=prop)
    
def plotLNlines(CL,CS,WS,WL,fs):
    MinLN = CL
    PpLN = max(0,CS - CL)
    CvLN = max(0,WS - MinLN - PpLN)
    PvLN = WL - MinLN - CvLN
    
    #MinLN
    addArrow([.6,3],[MinLN]*2)
    plt.plot([0.6]*2,[0,MinLN],'-',color='k',lw=4)
    plt.text(0.7,CL-2,'$MinLN=$'+"{:.1f}".format(MinLN),fontsize=fs,verticalalignment='center')
    #PpLN
    if PpLN>0.5:
        addArrow([1.6,2.6],[MinLN+PpLN]*2,'darkorange')
        plt.plot([1.6,1.6],[CS,CL],'-',color='darkorange',lw=4)
        plt.text(1.5,max((CL+CS)/2,CL+.75),'$PpLN=$'+"{:.1f}".format(PpLN),
                 fontsize=fs,verticalalignment='center',horizontalalignment='right',color='darkorange')
    else:
        plt.text(1.3,(CL+CS)/2,'$PpLN=0$',fontsize=fs,verticalalignment='bottom',color='darkorange')
    #CvLN
    if CvLN > 0.5:
        plt.plot([2.6,2.6],[CS,MinLN+PpLN+CvLN],'-',color='darkblue',lw=4)
        plt.text(2.5,max((CS+WS)/2,CS+.75),'$CvLN=$'+"{:.1f}".format(CvLN),
                 fontsize=fs, verticalalignment='center',horizontalalignment='right',color='darkblue')
        
        plt.plot([3,3],[MinLN,MinLN+CvLN],'-',color='darkblue',lw=4)
        addArrow([2.6,3],[MinLN+PpLN+CvLN*.5,MinLN+CvLN*.5],'darkblue')
        addArrow([3,3.2],[MinLN + CvLN]*2,'darkblue')
    else:
        plt.text(2.1,(CS+WS)/2,'$CvLN=0$',fontsize=fs, verticalalignment='bottom',color='darkblue')
    #PvLN
    if ((PvLN > 0.5)or(PvLN<-0.5)):
        plt.plot([3.2,3.2],[CL+CvLN,WL],'-',color='forestgreen',lw=4)
        plt.text(3.3,max(CL+CvLN + PvLN/2,WL+0.75),'$PvLN=$'+"{:.1f}".format(PvLN),
                 fontsize=fs, verticalalignment='center',color = 'darkgreen')
    else:
        plt.text(3.3,WL+0.75,'$PvLN=0$',fontsize=fs, verticalalignment='center',color = 'darkgreen')
    addArrow([3.6,3.2],[WL,WL],'darkgreen')
    
def ExpressionPlot(data,c,ylab=True,xlab=True):
    lowestTS = 16
    highestTS = 0
    pos = 0
    #Plot TS for all treats at apparent Vrn1 expression of 2.0
    for pv in ['WS','CL','CS','WL']:
        #plt.plot(CampInputs.loc[c,'FLN_'+pv],2.5,'*',mfc = fills[pos],mec=colors[pos],ms=10)
        plt.plot([data['TS_'+pv]],[2],'o',mfc = fills[pos],mec=colors[pos],ms=10)
        lowestTS = min(lowestTS,data['TS_'+pv])
        highestTS = max(highestTS,data['TS_'+pv])
        plt.plot([data['TS_'+pv]],[2.1],'v',color = 'k',ms=7)
        plt.plot([data['TS_'+pv]]*2,[2.1,2.2],'-',color = 'k',ms=7)
        pos+=1
    plt.plot([lowestTS,highestTS * 1.2],[2.2,2.2],'-',color = 'k',ms=7)
    plt.text(highestTS * 1.25, 2.2,'TS',horizontalalignment='center')
    
    #Plot VI_CL treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_CL']],[1],'o',mec='b',mfc='b',ms=10)
    plt.plot([data['VI_CL'],data['TS_CL']],[2,2],'--',color='k')
    plt.text((data['VI_CL']+data['TS_CL'])/2,2.05,'MinVI->TS',horizontalalignment='center')
    plt.plot([data['VI_CL']]*2,[1,2],'--',color='k')
    plt.text(data['VI_CL'],1.05,'VI_CL',horizontalalignment='center',verticalalignment='bottom')
    #Plot VI_LN treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_WL'],data['TS_WL']],[2,2],'--',color='k')
    plt.text((data['VI_WL']+data['TS_WL'])/2,2.05,'MinVI->TS',horizontalalignment='center')
    plt.plot([data['VI_WL']]*2,[1,2],'--',color='k')
    plt.plot([data['VI_WL']],[1],'o',mec='r',mfc='r',ms=10)
    plt.text(data['VI_WL'],1.05,'VI_WL',horizontalalignment='center',verticalalignment='bottom')
    #Plot VI_CS treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_CS']],[1],'o',mec='b',mfc='w',ms=10)
    plt.text(data['VI_CS'],0.95,'VI_CS',horizontalalignment='center',verticalalignment='top')
    plt.plot([data['VI_CS'],data['TS_CS']],[1.8,1.8],':',color='k')
    plt.plot([data['VI_CS'],data['VI_CS'],np.nan,data['TS_CS'],data['TS_CS']],[1,1.8,np.nan,1.8,2],':',color='k')
    plt.text((data['VI_CS']+data['TS_CS'])/2,1.85,'VI->Ts_S',horizontalalignment='center')
    #Plot VI_WS treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_WS']],[1],'o',mec='r',mfc='w',ms=10)
    plt.text(data['VI_WS'],0.95,'VI_WS',horizontalalignment='center',verticalalignment='top')
    plt.plot([data['VI_WS'],data['TS_WS']],[1.65,1.65],':',color='k')
    plt.plot([data['VI_WS'],data['VI_WS'],np.nan,data['TS_WS'],data['TS_WS']],[1,1.65,np.nan,1.65,2],':',color='k')
    plt.text((data['VI_WS']+data['TS_WS'])/2,1.7,'VI->Ts_S',horizontalalignment='center')
    
    #Plot base Vern1 Rate Vegetative
    plt.plot([-data['EmergDurat'],data['VI_WS']],[0,(data['VI_WS']+data['EmergDurat'])*data['rVrnBVeg']],'-',color='g')
    MidP = (-data['EmergDurat']+data['VI_WS'])/2
    plt.plot([MidP,MidP+1],[0.5,0.5],'--',color='g')
    plt.text(MidP+.5,0.53,'rVrnBVeg',color='g')
    
    #Plot max Vrn1 rate vegetative
    plt.plot([-data['EmergDurat'],data['VI_CS']],[0,(data['VI_CS']+data['EmergDurat'])*data['rVrnMVeg']],'-',color='gold')
    MidP = (data['VI_CL']+data['TS_CL'])/2
    plt.plot([MidP,MidP+1],[1.5,1.5],'-',color='gold')
    plt.text(MidP+.5,.53,'maxDVrnVeg',color='gold')
    
    #Plot base Vern1 Rate Early reproductive
    plt.plot([data['VI_WS'],data['TS_WS']],[1,1+(data['TS_WS'] - data['VI_WS'])*data['rVrnBER']],'-',color='g')
    MidP = (data['VI_WS']+data['TS_WS'])/2
    plt.plot([MidP,MidP+1],[1.5,1.5],'--',color='g')
    plt.text(MidP+.5,1.53,'rVrnBER',color='g')
    
    #Plot max Vrn1 ER
    plt.plot([data['VI_CL'],data['TS_CL']],[1,1+(data['TS_CL'] - data['VI_CL'])*data['rVrnMER']],'-',color='gold')
    MidP = (data['VI_CL']+data['TS_CL'])/2
    plt.plot([MidP,MidP+1],[1.5,1.5],'-',color='gold')
    plt.text(MidP+.5,1.53,'rVrnMER',color='gold')
    
    # Plot end of vernalisation timing
    data['endVern'] = min(data['EndVernTreat'],data['VI_CL'])
    plt.plot([data['endVern']]*2,[-1,2],'--',color='brown')
    plt.text(data['endVern'],-0.1,'endVern',horizontalalignment='center',color='brown')
    
    #Extrapolate vrn1 back from VI_WL to show when it started effective expression
    plt.plot([data['DRVrn2HSWL'],data['VI_WL']],[0,data['URVrn3HSWL'] * (data['rVrnBVeg'] * data['rVrn3Veg'])],'--',color='grey')
    MidP = data['DRVrn2HSWL']+data['URVrn3HSWL']*0.1
    plt.plot([MidP,MidP+1],[0.1]*2,'--',color='grey')
    plt.text(MidP+.2,0.15,'dvrnxVrn3',color='grey')
    
    #Extrapolate BaseVrn1 back to MinVI to show how much Vrn1 was required to overcome Vrn2
    plt.plot([0,data['DRVrn2HSWL']],[data['mVrn2'],0],'--',color='Orange')
    plt.text(0,min(2.15,data['mVrn2']),'Vrn2',color='orange')
    
    #The amount of Vrn1 required to reach VI is light is 1 + mVrn2
    plt.plot([0,data['VI_CL']],[1+data['mVrn2']]*2,'--',color='k')
    plt.text(data['VI_CL']-.2,1+data['mVrn2']+.03,'Vrn Target',horizontalalignment='right',verticalalignment='bottom')
    #the amount required from cold up regulation is the above less base Vrn1 until the end of Vrn2 less base * Vrn3 between end vrn2 and VI
    plt.plot([0,data['VI_CL']],[1+data['mVrn2']-data['VrnBVeg_CL']]*2,'-',color='g')
    plt.text(data['VI_CL']-.2,data['pVrn1_CL']+.03,'less baseVrn',horizontalalignment='right',verticalalignment='bottom')
    
    #RelMethTime = camp.CampConstants['MethalationThreshold']/(data['pVrn1_CL']+camp.CampConstants['MethalationThreshold'])
    StartMethHS = -data['EmergDurat'] + data['URVrn1HS_CL'] * 0.5 
    plt.plot([StartMethHS,data['endVern']],[0,data['pVrn1_CL']],'-',color='cyan')
    plt.text(-data['EmergDurat'],data['pVrn1_CL'],'pVrn1_CL',horizontalalignment='right',rotation=90,verticalalignment='top',color='c')
    
    if ylab==True:
        plt.ylabel('Apical Development index')
    if xlab==True:
        plt.xlabel('Haun Stage')
    plt.ylim(-.2,2.5)
    plt.xlim(-2,20)
    plt.text(19,0.05,c,horizontalalignment='right')
    
def plotVITS(data, level,ax, c,xmax=18, ylab=True,xlab=True):
    lowestTS = 16
    highestTS = 0
    pos = 0
    offset = 0.25
    #Plot TS for all treats at apparent Vrn1 expression of 2.0
    for pv in ['WS','CL','CS','WL']:
        #plt.plot(CampInputs.loc[c,'FLN_'+pv],2.5,'*',mfc = fills[pos],mec=colors[pos],ms=10)
        plt.plot([data['TS_'+pv]],[2],'o',mfc = fills[pos],mec=colors[pos],ms=10)
        plt.text(data['TS_'+pv],2+offset,pv,horizontalalignment='center')
        lowestTS = min(lowestTS,data['TS_'+pv])
        highestTS = max(highestTS,data['TS_'+pv])
        plt.plot([data['TS_'+pv]],[2.1],'v',color = 'k',ms=7)
        plt.plot([data['TS_'+pv]]*2,[2.1,1.95+offset],':',color = 'k',ms=7)
        pos+=1
        offset +=0.15
    #plt.text(-1, 2.05,r'$TS$',horizontalalignment='center',verticalalignment='center')
    plt.plot([-2,20],[2,2],ls='dotted',color='k')
    if ylab == True:
        plt.ylabel('Apical Development index')
    if xlab == True:
        plt.xlabel('Haun Stage')
    plt.ylim(0,3)
    plt.xlim(-2,xmax)
    ax.set_yticks([0,0.5,1,1.5,2,2.5,3],['0.0','0.5','VI','1.5','TS','2.5','3.0'])
    plt.text(xmax*.95,3*.95,c,horizontalalignment='right')
    if(level>1):
        #plt.text(-1, 1.05,r'$VI$',horizontalalignment='center',verticalalignment='center')
        plt.plot([-2,20],[1,1],ls='dotted',color='k')
        #Plot VI_CL treatments at apparent Vrn1 expression of 1.0
        plt.plot([data['VI_CL']],[1],'o',mec='b',mfc='b',ms=10)
        plt.plot([data['VI_CL']+.1,data['TS_CL']-.1],[1.75,1.75],'>--',color='k')
        #plt.text((data['VI_CL']+data['TS_CL'])/2,1.75,r'$MinER^{HS}$',horizontalalignment='center',verticalalignment='bottom')
        plt.plot([data['VI_CL'],data['VI_CL'],np.nan,data['TS_CL'],data['TS_CL']],[1,1.75,np.nan,1.75,2],ls='dotted',color='k')
        plt.text(data['VI_CL'],.67,'CL',horizontalalignment='center',verticalalignment='bottom')
        plt.plot([data['VI_CL']],[0.9],'^',color = 'k',ms=7)
        plt.plot([data['VI_CL']]*2,[0.9,0.8],':',color = 'k',ms=7)
       #Plot VI_LN treatments at apparent Vrn1 expression of 1.0
        plt.plot([data['VI_WL']+.1,data['TS_WL']-.1],[1.5,1.5],'>--',color='k')
        plt.text((data['VI_WL']+data['TS_WL'])/2,1.5,r'$MinER^{HS}$',horizontalalignment='center',verticalalignment='bottom',color='k')
        plt.plot([data['VI_WL'],data['VI_WL'],np.nan,data['TS_WL'],data['TS_WL']],[1,1.55,np.nan,1.55,2],ls='dotted',color='k')
        plt.plot([data['VI_WL']],[1],'o',mec='r',mfc='r',ms=10)
        plt.text(data['VI_WL'],.3,'WL',horizontalalignment='center',verticalalignment='bottom')
        plt.plot([data['VI_WL']],[0.9],'^',color = 'k',ms=7)
        plt.plot([data['VI_WL']]*2,[0.9,0.45],':',color = 'k',ms=7)
        if(level>2):    
            #Plot VI_CS treatments at apparent Vrn1 expression of 1.0
            plt.plot([data['VI_CS']],[1],'o',mec='b',mfc='w',ms=10)
            plt.text(data['VI_CS'],0.67,'CS',horizontalalignment='center',verticalalignment='top')
            plt.plot([data['VI_CS']+.1,data['TS_CS']-.1],[1.4,1.4],'>-.',color='grey')
            plt.plot([data['VI_CS'],data['VI_CS'],np.nan,data['TS_CS'],data['TS_CS']],[1,1.4,np.nan,1.4,2],':',color='k')
            #plt.text((data['VI_CS']+data['TS_CS'])/2,1.35,r'$MaxER^{HS}$',horizontalalignment='center',verticalalignment='bottom')
            #Plot VI_WS treatments at apparent Vrn1 expression of 1.0
            plt.plot([data['VI_WS']],[1],'o',mec='r',mfc='w',ms=10)
            plt.text(data['VI_WS'],0.3,'WS',horizontalalignment='center',verticalalignment='top')
            plt.plot([data['VI_WS']+.1,data['TS_WS']-.1],[1.15,1.15],'>-.',color='grey')
            plt.plot([data['VI_WS'],data['VI_WS'],np.nan,data['TS_WS'],data['TS_WS']],[1,1.15,np.nan,1.15,2],':',color='k')
            plt.text((data['VI_WS']+data['TS_WS'])/2,1.15,r'$MaxER^{HS}$',horizontalalignment='center',verticalalignment='bottom',color='grey')
            plt.plot([data['VI_WS']],[0.9],'^',color = 'k',ms=7)
            plt.plot([data['VI_WS']]*2,[0.9,0.35],':',color = 'k',ms=7)
            if(level>3):
                #Plot base Vern1 Rate Vegetative
                plt.plot([-data['EmergDurat'],data['VI_WS']],[0,(data['VI_WS']+data['EmergDurat'])*data['rVrnBVeg']],'-',color='g')
                MidP = (-data['EmergDurat']+data['VI_WS'])/2
                plt.plot([MidP,MidP+1],[0.5,0.5],'--',color='g')
                plt.text(MidP+.5,0.53,'rVrnBVeg',color='g')
                #Plot max Vrn1 rate vegetative
                plt.plot([-data['EmergDurat'],data['VI_CS']],[0,(data['VI_CS']+data['EmergDurat'])*data['MaxDVrnVeg']],'-',color='gold')
                MidP = (data['VI_CL']+data['TS_CL'])/2
                plt.plot([MidP,MidP+1],[1.5,1.5],'-',color='gold')
                plt.text(MidP+.5,1.53,'maxDVrnER',color='gold')
                #Plot base Vern1 Rate Early reproductive
                plt.plot([data['VI_WS'],data['TS_WS']],[1,1+(data['TS_WS'] - data['VI_WS'])*data['rVrnBER']],'-',color='g')
                MidP = (data['VI_WS']+data['TS_WS'])/2
                plt.plot([MidP,MidP+1],[1.5,1.5],'--',color='g')
                plt.text(MidP+.5,1.53,'rVrnBER',color='g')

                #Plot max Vrn1 ER
                plt.plot([data['VI_CL'],data['TS_CL']],[1,1+(data['TS_CL'] - data['VI_CL'])*data['MaxDVrnER']],'-',color='gold')
                MidP = (data['VI_CL']+data['TS_CL'])/2
                plt.plot([MidP,MidP+1],[1.5,1.5],'-',color='gold')
                plt.text(MidP+.5,1.53,'MaxDVrnER',color='gold')
                if(level>4):
                    # Plot end of vernalisation timing
                    data['endVern'] = min(data['EndVernTreat'],data['VI_CL'])
                    plt.plot([data['endVern']]*2,[-1,2],'--',color='brown')
                    plt.text(data['endVern'],-0.1,'endVern',horizontalalignment='center',color='brown')

                    #Extrapolate vrn1 back from VI_WL to show when it started effective expression
                    plt.plot([data['endVrn2_WL'],data['VI_WL']],[0,data['vrnxVrn3Durat'] * data['dvrnxVrn3']],'--',color='grey')
                    MidP = data['endVrn2_WL']+data['vrnxVrn3Durat']*0.1
                    plt.plot([MidP,MidP+1],[0.1]*2,'--',color='grey')
                    plt.text(MidP+.2,0.15,'dvrnxVrn3',color='grey')

                    #Extrapolate BaseVrn1 back to MinVI to show how much Vrn1 was required to overcome Vrn2
                    plt.plot([0,data['endVrn2_WL']],[data['mVrn2'],0],'--',color='Orange')
                    plt.text(0,min(2.15,data['mVrn2']),'Vrn2',color='orange')

                    #The amount of Vrn1 required to reach VI is light is 1 + mVrn2
                    plt.plot([0,data['VI_CL']],[1+data['mVrn2']]*2,'--',color='k')
                    plt.text(data['VI_CL']-.2,1+data['mVrn2']+.03,'Vrn Target',horizontalalignment='right',verticalalignment='bottom')
                    #the amount required from cold up regulation is the above less base Vrn1 until the end of Vrn2 less base * Vrn3 between end vrn2 and VI
                    plt.plot([0,data['VI_CL']],[1+data['mVrn2']-data['VrnBVeg_CL']]*2,'-',color='g')
                    plt.text(data['VI_CL']-.2,data['pVrn1_CL']+.03,'less baseVrn',horizontalalignment='right',verticalalignment='bottom')

                    #RelMethTime = camp.CampConstants['MethalationThreshold']/(data['pVrn1_CL']+camp.CampConstants['MethalationThreshold'])
                    StartMethHS = -data['EmergDurat'] + data['Vrn1HS_CL'] * 0.5 
                    plt.plot([StartMethHS,data['endVern']],[0,data['pVrn1_CL']],'-',color='cyan')
                    plt.text(-data['EmergDurat'],data['pVrn1_CL'],'pVrn1_CL',horizontalalignment='right',rotation=90,verticalalignment='top',color='c')
    
def boundPlots(data, ax, c, xmax=18, ylab=True,xlab=True):
    lowestTS = 16
    highestTS = 0
    pos = 0
    
    #Plot TS for all treats at apparent Vrn1 expression of 2.0
    offset=0.25
    for pv in ['WS','CL','CS','WL']:
        #plt.plot(CampInputs.loc[c,'FLN_'+pv],2.5,'*',mfc = fills[pos],mec=colors[pos],ms=10)
        plt.plot([data['TS_'+pv]],[2],'o',mfc = fills[pos],mec=colors[pos],ms=10)
        plt.text(data['TS_'+pv],2+offset,pv,horizontalalignment='center')
        lowestTS = min(lowestTS,data['TS_'+pv])
        highestTS = max(highestTS,data['TS_'+pv])
        plt.plot([data['TS_'+pv]],[2.1],'v',color = 'k',ms=7)
        plt.plot([data['TS_'+pv]]*2,[2.1,1.95+offset],':',color = 'k',ms=7)
        offset+=0.15
        pos+=1
    #plt.text(-1, 2.05,r'$TS$',horizontalalignment='center',verticalalignment='center')
    plt.plot([-2,20],[2,2],ls='dotted',color='k')
    
    if ylab == True:
        plt.ylabel('Apical Development index')
    if xlab == True:
        plt.xlabel('Haun Stage')
    plt.ylim(0,3)
    plt.xlim(-2,xmax)
    ax.set_yticks([0,0.5,1,1.5,2,2.5,3],['0.0','0.5','VI','1.5','TS','2.5','3.0'])
    plt.text(xmax*.95,3*.95,c,horizontalalignment='right')
    
    #plt.text(-1, 1.05,r'$VI$',horizontalalignment='center',verticalalignment='center')
    plt.plot([-2,20],[1,1],ls='dotted',color='k')
    #Plot VI_CL treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_CL']],[1],'o',mec='b',mfc='b',ms=10)
    plt.text(data['VI_CL'],1.3,'CL',horizontalalignment='center',verticalalignment='bottom')
    plt.plot([data['VI_CL']],[1.1],'v',color = 'k',ms=7)
    plt.plot([data['VI_CL']]*2,[1.1,1.2],':',color = 'k',ms=7)
   #Plot VI_WL treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_WL']],[1],'o',mec='r',mfc='r',ms=10)
    plt.text(data['VI_WL'],.67,'WL',horizontalalignment='center',verticalalignment='bottom')
    plt.plot([data['VI_WL']],[0.9],'^',color = 'k',ms=7)
    plt.plot([data['VI_WL']]*2,[0.9,0.8],':',color = 'k',ms=7)
    
    #Plot VI_CS treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_CS']],[1],'o',mec='b',mfc='w',ms=10)
    plt.text(data['VI_CS'],1.3,'CS',horizontalalignment='center',verticalalignment='top')
    
    #Plot VI_WS treatments at apparent Vrn1 expression of 1.0
    plt.plot([data['VI_WS']],[1],'o',mec='r',mfc='w',ms=10)
    plt.text(data['VI_WS'],0.67,'WS',horizontalalignment='center',verticalalignment='top')
    # plt.plot([data['VI_WS']+.1,data['TS_WS']-.1],[1.15,1.15],'>-.',color='k')
    # plt.plot([data['VI_WS'],data['VI_WS'],np.nan,data['TS_WS'],data['TS_WS']],[1,1.15,np.nan,1.15,2],':',color='k')
    # plt.text((data['VI_WS']+data['TS_WS'])/2,1.15,r'$MaxER^{HS}$',horizontalalignment='center',verticalalignment='bottom')
    plt.plot([data['VI_WS']],[0.9],'^',color = 'k',ms=7)
    plt.plot([data['VI_WS']]*2,[0.9,0.7],':',color = 'k',ms=7)
            
    basepos = 0.25
    #Plot base Vern1 Rate Vegetative
    plt.plot([-data['EmergDurat'],data['VI_WS']],[0,(data['VI_WS']+data['EmergDurat'])*data['rVrnBVeg']],'-',color='g')
    MidP = ((data['EmergDurat']+data['VI_WS'])* basepos)-data['EmergDurat']
    ang = np.arctan(1/(data['EmergDurat']+data['VI_WS']))*(180/np.pi)
    arc = patches.Arc([MidP,basepos],1.0,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='g')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[basepos]*2,'-',color='g')
    plt.text(MidP+.7,basepos+.03,r'$rVrnB_{Veg}$',color='g')
    
    maxpos = 0.75
    #Plot max Vrn1 rate vegetative
    plt.plot([-data['EmergDurat'],data['VI_CS']],[0,(data['EmergDurat']+data['VI_CS'])*data['rVrnMVeg']],'-',color='gold')
    MidP = ((data['EmergDurat']+data['VI_CS']) * maxpos)-data['EmergDurat']
    ang = np.arctan(1/(data['EmergDurat']+data['VI_CS']))*(180/np.pi)
    arc = patches.Arc([MidP,maxpos],1.0,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='gold')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[maxpos]*2,'-',color='gold')
    plt.text(MidP+.7,maxpos+0.03,r'$rVrnM_{Veg}$',color='gold', horizontalalignment = 'left')
    
    #Plot base Vern1 Rate Early reproductive
    plt.plot([data['VI_WS'],data['TS_WS']],[1,1+(data['TS_WS'] - data['VI_WS'])*data['rVrnBER']],'-',color='g')
    MidP =  data['VI_WS'] + (data['VI->TS_CS'] * basepos)
    ang = np.arctan(1/data['VI->TS_CS'])*(180/np.pi)
    arc = patches.Arc([MidP,1+basepos],1.0,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='g')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[1+basepos]*2,'-',color='g')
    plt.text(MidP+.7,1+basepos+.03,r'$rVrnB_{ER}$',color='g')

    # #Plot max Vrn1 ER
    plt.plot([data['VI_CL'],data['TS_CL']],[1,1+(data['TS_CL'] - data['VI_CL'])*data['rVrnMER']],'-',color='gold')
    MidP = data['VI_CL']+(data['MinVI->TS']* maxpos)
    ang = np.arctan(1/(data['EmergDurat']+data['VI_CS']))*(180/np.pi)
    arc = patches.Arc([MidP,1+maxpos],1.0,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='gold')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[1+maxpos]*2,'-',color='gold')
    plt.text(MidP+.7,1+maxpos+0.03,r'$rVrnM_{ER}$',color='gold',)
    
    plotImbibEmerg(data)
    
def Vrn1Plots(data,ax, c,xmax=10,ymax=3, ylab=True, xlab=True, figY_X = 1):
    lowestTS = 16
    highestTS = 0
    ymin = 0
    xmin = -1.2
    yrange = ymax-ymin
    xrange = xmax-xmin
    pos = 0

    if ylab == True:
        plt.ylabel('Apical Development index')
    if xlab == True:
        plt.xlabel('Haun Stage')
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    yposs,ylabs = makeTickBits(ymax)
    ax.set_yticks(yposs,ylabs)
    plt.text(xmax*.95,ymax*.95,c,horizontalalignment='right')
    
    # Plot VI and mark up
    plt.plot([data['VI_CS']],[1],'o',mec='b',mfc='b',ms=10)
    
    plt.text(data['VI_CL'],-0.02,r'$VI_{CL}$',horizontalalignment='center',verticalalignment='top')
    plt.plot([-2,20],[1,1],ls='dotted',color='k')
    plt.plot([data['VI_CL']],[0.9],'^',color = 'k',ms=7)
    plt.plot([data['VI_CL']]*2,[0.9,data['VrnBVeg_CL']],':',color = 'k',ms=7)
 
    #plot mVrn2 on top of VI to show how much Vrn1 was required to overcome Vrn2
    plt.plot([-2,xmax*0.95],[data['VrnPVI_CL']]*2,'--',color='k')
    
    plt.text(xmax*0.95,1+data['mVrn2'],r'$mVrn2$',color='crimson',horizontalalignment='right',verticalalignment='bottom')
    plt.plot([xmax*0.95]*2,[1.02,0.98+data['mVrn2']], '^--', color='crimson')
    plt.text(-1,1.02+data['mVrn2'],r'$Vrn^PVI_{CL}$')
    
    #Plot base Vern1 Rate Vegetative
    plt.plot([-data['EmergDurat'],data['VI_CL']],[0,data['VrnBVeg_CL']],'-',color='g')
    
    basepos = 0.12
    MidP = - data['EmergDurat'] + basepos/data['rVrnBVeg']
    ang = math.atan((data['VrnBVeg_CL'])/(data['EmergDurat']+data['VI_CL']))*(180/np.pi)
    arc = patches.Arc([MidP,basepos],1.0,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='g')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[basepos]*2,'-',color='g')
    plt.text(MidP,basepos,r'$rVrnB_{Veg}$',color='g',horizontalalignment='left',verticalalignment='top')
    plt.plot([data['VI_CL']]*2,[0.02,data['VrnBVeg_CL']-.02],'^-.',color='g')
    #plt.text(data['VI_CL']+.02,0.2,r'$\Sigma^0_{VI}VrnB_{Veg}$',horizontalalignment='left',verticalalignment='bottom',color='g')
    
    # plot sigmaVrnBveg below total Vrn1 requirement to show how much cold up reg is required
    plt.plot([-2,xmax*0.85],[data['pVrn1_CL']]*2,'-.',color='k')
    
    yposVrnBVegupper = 0.98+data['mVrn2']
    plt.plot([xmax*0.85]*2,[data['pVrn1_CL']+.02,data['VrnPVI_CL']-.02],'^-.',color='g')
    plt.text(xmax*0.85,yposVrnBVegupper,r'$\Sigma^0_{VI}VrnB_{Veg}$',horizontalalignment='right',verticalalignment='top',color='g')
    plt.text(-1,1.02+data['mVrn2']-data['VrnBVeg_CL'],'$pVrn1_{CL}$')
    #plt.plot([data['VI_CL']+.7,data['VI_CL']+.7,data['VI_CL']+.5],[0.35,yposVrnBVegupper,yposVrnBVegupper],'-.',color='g')
    #plt.plot(data['VI_CL']+.45,yposVrnBVegupper,'<',color='g')
    
    # plot line to show end of cold treatment
    plt.plot([data['EndVernTreat']]*2,[yrange*.15,0],':',color = 'b',ms=7)
    plt.plot(data['EndVernTreat'],0.03,'v',mec = 'b',mfc='b',ms=7)
    plt.text(data['EndVernTreat']+xrange*.04,0,'End\nCold\ntrt', rotation = 0, verticalalignment='bottom',horizontalalignment = 'left',color='b')
    
    # Plot line to show persistant Vrn1 expression
    MethHS = data['URVrn1HS_CL']*.5 - data['EmergDurat']
    xsMeth = np.array((MethHS,MethHS+data['URVrn1HS_CL']*.5))
    ysMeth = np.array((0,data['pVrn1_CL']))
    plt.plot(xsMeth,ysMeth,'-',color='b') 
    
    basepos = data['pVrn1_CL'] * .5
    MidP = data['URVrn1HS_CL'] * 0.75 - data['EmergDurat']
    ang = math.atan(data['pVrn1_CL']/(data['URVrn1HS_CL']*.5))*(180/np.pi)
    arc = patches.Arc([MidP,basepos],0.5,.125,angle=0.0,theta1=0.0,theta2=ang,linewidth=2, fill=False, color='b')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+MidP*xrange*0.1],[basepos,basepos],'-',color='b')
    plt.text(MidP,basepos,r'$\Delta Vrn1_{CL}$',color='b', verticalalignment='top',horizontalalignment = 'left')
    trot = calcTextRotation(xsMeth,ysMeth,xrange,yrange,figY_X)
    plt.text(np.mean(xsMeth),np.mean(ysMeth)+0.05*yrange,'meth Vrn1', rotation = trot, verticalalignment='center',horizontalalignment = 'center',color='b')
    
    #plot line to show unmethalated Vrn1 expression.
    plt.plot([-data['EmergDurat'],MethHS],[0,data['pVrn1_CL']],'-',color='deepskyblue')
    plt.text((-data['EmergDurat']+MethHS)/2-(xrange*.05),np.mean(ysMeth),'unmeth Vrn1',
                rotation=trot, verticalalignment='center',horizontalalignment = 'center',color='deepskyblue')
    
def Vrn2Plots(data, ax, c, xmax=18, ymax=3, ylab=True, xlab=True):
    lowestTS = 16
    highestTS = 0
    pos = 0

    if ylab == True:
        plt.ylabel('Apical Development index')
    if xlab == True:
        plt.xlabel('Haun Stage')
    plt.ylim(0,ymax)
    plt.xlim(-2,xmax)
    yposs,ylabs = makeTickBits(ymax)
    ax.set_yticks(yposs,ylabs)
    plt.text(xmax*.95,ymax*.95,c,horizontalalignment='right')
    
    #plt.text(-1.7, 1.05,r'$VI$',horizontalalignment='center',verticalalignment='center')
    plt.plot([-2,20],[1,1],ls='dotted',color='k')

    plt.plot([data['VI_WL']],[1],'o',mec='r',mfc='r',ms=10)
    plt.text(data['VI_WL'],.67,'WL',horizontalalignment='center',verticalalignment='bottom')
    plt.plot([data['VI_WL']],[0.9],'^',color = 'k',ms=7)
    plt.plot([data['VI_WL']]*2,[0.9,0.8],'-',color = 'k',ms=7)
 
    #Extrapolate BaseVrn1 back to imbibition to show how much Vrn1 was required to overcome Vrn2
    plt.plot([-data['EmergDurat'],data['DRVrn2HSWL']],[data['mVrn2'],0],'--',color='g')
    plt.text(0,data['mVrn2'],R'$mVrn2$',color='crimson',horizontalalignment='left',verticalalignment='bottom')
    plt.plot([-data['EmergDurat']-.5,data['DRVrn2HSWL']],[data['mVrn2']]*2, '-', color='crimson')
    plt.plot([-data['EmergDurat']-.5]*2,[0.03,data['mVrn2']-0.03],'^--',color='crimson')
        
    # #Plot max Vrn1 ER
    maxpos = 0.5
    MidP = ((data['DRVrn2HSWL'] - data['EmergDurat'])* maxpos)
    ang = math.atan(data['mVrn2']/(data['DRVrn2HSWL']+data['EmergDurat']))*(180/np.pi)
    arc = patches.Arc([MidP,data['mVrn2']*(1-maxpos)],1.0,.125,angle=0.0,theta1=360-ang,theta2=0,linewidth=2, fill=False, color='g')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[data['mVrn2']*(1-maxpos)]*2,'-',color='g')
    plt.text(MidP+.6,data['mVrn2']*(1-maxpos),r'$-(rVrnB_{Veg})$',color='g',horizontalalignment='left',verticalalignment='top')

    plt.plot([data['DRVrn2HSWL'],data['VI_WL']],[0,data['URVrn3HSWL']*(data['rVrnBVeg'] * data['rVrn3Veg'])],'--',color='grey')
    maxpos = 0.5
    MidP = data['DRVrn2HSWL']+(data['VI_WL']-data['DRVrn2HSWL'])* maxpos
    ang = math.atan(1/(data['VI_WL']-data['DRVrn2HSWL']))*(180/np.pi)
    arc = patches.Arc([MidP,1*maxpos],1.0,.125,angle=0.0,theta1=0,theta2=ang,linewidth=2, fill=False, color='grey')
    ax.add_patch(arc)
    plt.plot([MidP,MidP+1],[maxpos]*2,'-',color='grey')
    plt.text(MidP+.6,maxpos+0.03,r'$rVrnB_{Veg} * rVrn3_{Veg}$',color='grey',)
    plt.plot([data['VI_WL']]*2, [1.0,1.15],':',color='k')
    plt.plot([data['DRVrn2HSWL']+.1,data['DRVrn2HSWL']+data['URVrn3HSWL']],[1.15,1.15],'>-.',color='grey')
    plt.text(data['DRVrn2HSWL'],1.15,r'$URVrn3^{HS}_{WL}$',horizontalalignment='left',verticalalignment='bottom',color='grey')
    
    plotImbibEmerg(data)
  
    if (data['DRVrn2HSWL']>0.1):
        plt.plot([0+.1,data['DRVrn2HSWL']],[1.15,1.15],'>-.',color='k')
        plt.plot([0,0,np.nan,data['DRVrn2HSWL'],data['DRVrn2HSWL']],[1,1.15,np.nan,1.15,0],':',color='k')
        plt.text(data['DRVrn2HSWL'],1.15,r'$DRVrn2^{HS}_{WL}$',horizontalalignment='right',verticalalignment='bottom',color='k')      
                           
def calcTextRotation(xs,ys,xrange,yrange,figY_X):
    dx = xs[1] - xs[0] 
    dy = ys[1] - ys[0]
    Dx = dx  / xrange
    Dy = dy  / yrange * figY_X
    return (180/np.pi)*np.arctan(Dy/Dx)

def plotImbibEmerg(data):
    plt.plot(0,0.03,'v',mec = 'k',mfc='white',ms=7)
    plt.plot([0,0],[.06,1.42],':',color = 'k',ms=7)
    plt.text(0,1.45,'Emergence', rotation = 90, verticalalignment='bottom',horizontalalignment = 'center')
    
    plt.plot(-data['EmergDurat'],0.03,'v',color = 'k',ms=7)
    plt.plot([-data['EmergDurat']]*2,[0,1.42],'-',color = 'k',ms=7)
    plt.text(-data['EmergDurat'],1.45,'Imbibation', rotation = 90, verticalalignment='bottom',horizontalalignment = 'center')
    
def makeTickBits(ymax):
    ys = np.arange(0,ymax+0.01,0.5)
    tickLabs = []
    for y in ys:
        val = str(y)
        if y == 1.0:
            val = 'VI'
        if y == 2.0:
            val = 'TS'
        tickLabs.append(val)
    return ys, tickLabs

# %%
