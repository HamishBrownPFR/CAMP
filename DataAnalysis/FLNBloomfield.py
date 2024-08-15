# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MathsUtilities as MUte
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.dates as mdates
import CAMP as camp
import copy
import json
import math
# %matplotlib inline

Envs = ['WL', 'CL', 'WS', 'CS']

EnvsExptNames = ['LN','LV','SN','SV']

EnvsNameConvert = pd.Series(index=EnvsExptNames,data=Envs)


def convertVernTreatnames(N):
    if N == 'V':
        return 'C'
    if N == 'N':
        return 'W'


EnvsNameConvert

SowDates = pd.Series(index = Envs,data = ['2018-11-03','2018-11-03','2018-07-06','2018-08-04'])
SowDates

TransitionDates =  pd.Series(index = Envs,data = ['2018-11-04','2019-01-04','2018-07-07','2018-09-28'])
TransitionDates

# ## Bring in trial design data frame

Design = pd.read_excel('./RawData/NPI_Agribio2018/expdb_AgriBio2018.xlsx',sheet_name='Design',engine='openpyxl')
Design.set_index('Plot',inplace=True)

# ## Bring in MetData to calculate thermal time

# +
#Bring in the APSIM met files
WLmet = pd.read_csv('./RawData/NPI_Agribio2018/LN.met',sep='\t',skiprows=11)
WLmet.drop(0,inplace=True)
for c in WLmet.columns:
    WLmet.loc[:,c] = pd.to_numeric(WLmet.loc[:,c])
WLmet.loc[:,'date'] = [dt.datetime(WLmet.loc[x,'year'],1,1) + dt.timedelta(float(WLmet.loc[x,'day'])) for x in WLmet.index]
WLmet.set_index('date',inplace=True)
WLmet.drop(['year','day'],axis=1,inplace=True)
WLmet.loc[SowDates['WL']:,'AccumTtSow'] = WLmet.loc[SowDates['WL']:,'mint'].cumsum()

WSmet = pd.read_csv('./RawData/NPI_Agribio2018/SN.met',sep='\t',skiprows=11)
WSmet.drop(0,inplace=True)
for c in WSmet.columns:
    WSmet.loc[:,c] = pd.to_numeric(WSmet.loc[:,c])
WSmet.loc[:,'date'] = [dt.datetime(WSmet.loc[x,'year'],1,1) + dt.timedelta(float(WSmet.loc[x,'day'])) for x in WSmet.index]
WSmet.set_index('date',inplace=True)
WSmet.drop(['year','day'],axis=1,inplace=True)
WSmet.loc[SowDates['WS']:,'AccumTtSow'] = WSmet.loc[SowDates['WS']:,'mint'].cumsum()


CLmet = pd.read_csv('./RawData/NPI_Agribio2018/LV.met',sep='\t',skiprows=11)
CLmet.drop(0,inplace=True)
for c in CLmet.columns:
    CLmet.loc[:,c] = pd.to_numeric(CLmet.loc[:,c])
CLmet.loc[:,'date'] = [dt.datetime(CLmet.loc[x,'year'],1,1) + dt.timedelta(float(CLmet.loc[x,'day'])) for x in CLmet.index]
CLmet.set_index('date',inplace=True)
CLmet.drop(['year','day'],axis=1,inplace=True)
CLmet.loc[SowDates['CL']:,'AccumTtSow'] = CLmet.loc[SowDates['CL']:,'mint'].cumsum()

CSmet = pd.read_csv('./RawData/NPI_Agribio2018/SV.met',sep='\t',skiprows=11)
CSmet.drop(0,inplace=True)
for c in CSmet.columns:
    CSmet.loc[:,c] = pd.to_numeric(CSmet.loc[:,c])
CSmet.loc[:,'date'] = [dt.datetime(CSmet.loc[x,'year'],1,1) + dt.timedelta(float(CSmet.loc[x,'day'])) for x in CSmet.index]
CSmet.set_index('date',inplace=True)
CSmet.drop(['year','day'],axis=1,inplace=True)
CSmet.loc[SowDates['CS']:,'AccumTtSow'] = CSmet.loc[SowDates['CS']:,'mint'].cumsum()

Met = pd.concat([WSmet,WLmet,CSmet,CLmet],keys=['WS','WL','CS','CL'])

#Bring in the datalogger data from in the growth rooms
Temp = pd.read_excel('./RawData/NPI_Agribio2018/GrowthRoomTemp.xlsx',engine='openpyxl')
Temp.Date = pd.to_datetime(Temp.Date)
Temp.Environment = [EnvsNameConvert[Temp.loc[x,'Environment']] for x in Temp.index]
Temp.set_index(['Environment','Date'],inplace=True)
CanopyTemp = pd.read_excel('./RawData/NPI_Agribio2018/GrowthRoomsCanopyTemp.xlsx',header=2,engine='openpyxl')
CanopyTemp.Date = pd.to_datetime(CanopyTemp.Date)
CanopyTemp.Environment = [EnvsNameConvert[CanopyTemp.loc[x,'Environment']] for x in CanopyTemp.index]
CanopyTemp.set_index(['Environment','Date'],inplace=True)
# -

# ## Graph up met data to check it looks right

Graph = plt.figure(figsize=(18,20))
pos=1
for e in Envs:
    ax = Graph.add_subplot(4,1,pos)
    SowDate = pd.to_datetime(SowDates[e])
    ChangeDate = pd.to_datetime(TransitionDates[e])
    StartSlice = SowDate + dt.timedelta(110)
    EndSlice = StartSlice + dt.timedelta(7)
    plt.plot(Temp.loc[e,'AirTemp'].loc[SowDate:],label='AirTemp')
    plt.plot(Temp.loc[e,'SoilTemp'].loc[SowDate:],label='SoilTemp')
    plt.plot(CanopyTemp.loc[e,'LeafTemp'].loc[SowDate:],color='g',label='LeafTemp')
    plt.plot(Met.loc[e,'mint'].loc[SowDate:],label='minT')
    plt.plot(Met.loc[e,'mint'].loc[SowDate:],label='maxT')
    plt.plot([ChangeDate]*2,[0,30],color='k',label='60 days')
    plt.plot([SowDate]*2,[0,30],'--',color='k',label='Sowing')
    ax.axvspan(StartSlice,EndSlice,alpha=0.2,color='r')
    plt.text(0.05,0.9,e,transform = ax.transAxes,fontsize=24)
    plt.ylim(0,30)
    plt.legend(loc=4,fontsize=16)
    plt.ylabel('Temperature (oC)',fontsize=20)
    plt.tick_params(labelsize=16)
    pos+=1
Graph.patch.set_facecolor('white')

# ### The pink shaded period is expanded in the plot below

Graph = plt.figure(figsize=(18,20))
pos=1
for e in Envs:
    ax = Graph.add_subplot(4,1,pos)
    SowDate = pd.to_datetime(SowDates[e])
    ChangeDate = SowDate + dt.timedelta(60)
    StartSlice = SowDate + dt.timedelta(110)
    EndSlice = StartSlice + dt.timedelta(7)
    plt.plot(Temp.loc[e,'AirTemp'].loc[StartSlice:EndSlice],label='AirTemp')
    plt.plot(Temp.loc[e,'SoilTemp'].loc[StartSlice:EndSlice],label='SoilTemp')
    plt.plot(CanopyTemp.loc[e,'LeafTemp'].loc[StartSlice:EndSlice],color='g',label='LeafTemp')
    plt.plot(Met.loc[e,'mint'].loc[StartSlice:EndSlice],label='minT')
    plt.plot(Met.loc[e,'mint'].loc[StartSlice:EndSlice],label='maxT')
    plt.text(0.05,0.9,e,transform = ax.transAxes,fontsize=24)
    plt.ylim(0,30)
    plt.legend(loc=4,fontsize=16)
    plt.ylabel('Temperature (oC)',fontsize=20)
    plt.tick_params(labelsize=16)
    pos+=1
Graph.patch.set_facecolor('white')

# ## Points to note
# - Duration of vernalisation treatment was less for the SV treatment
# - Air, soil and leaf tempertures appear close together and the temperaures we have in the APSIM file do a reasonable job of approximating this.

# ## Next lets bring in the FLN data

FLNData = pd.read_excel('./RawData/NPI_Agribio2018/expdb_AgriBio2018.xlsx',sheet_name='Phenotype_FLN',engine='openpyxl')
FLNData.loc[:,'Photoperiod'] = [Design.loc[x,'TrialCode'].split('-')[1][0] for x in FLNData.loc[:,'Plot']]
FLNData.loc[:,'VegPhaseTemp'] = [Design.loc[x,'TrialCode'].split('-')[1][1] for x in FLNData.loc[:,'Plot']]
FLNData.loc[:,'VegPhaseTemp'] = [convertVernTreatnames(FLNData.loc[x,'VegPhaseTemp']) for x in FLNData.index]
FLNData.loc[:,'Genotype'] = [Design.loc[x,'Genotype'] for x in FLNData.loc[:,'Plot']]
FLNData.set_index(['Photoperiod','VegPhaseTemp','Genotype'],inplace=True)
FLNData.sort_index(axis=0,inplace=True)
Cultivars = FLNData.index.get_level_values(2).drop_duplicates()


def MakeGenotypeBarPlot(Geno,Axis):
    width = 0.4
    ind = np.arange(4) + width
    colors = ['b','b','r','r']
    fills = ['b','w','w','r']
    SubPos = 0
    for Photo,VegT in [('L','C'),('S','C'),('S','W'),('L','W')]:
            try:
                y = FLNData.loc[(Photo,VegT,Geno),'Value']
            except:
                y = []
            try:
                x = np.add(list(range(len(y))),SubPos)
            except:
                x = []
            plt.plot(x,y, 'o', markersize = 12, mec = colors[SubPos],
                     mfc=fills[SubPos], label = Photo + " " + VegT)
            try:
                plt.bar(SubPos,y.median(),0.8,
                        edgecolor=colors[SubPos], color = fills[SubPos],
                        linewidth=3,alpha=0.4)
            except:
                dummy = 'bla bla'
            #plt.xticks([3,9,15,21])
            plt.text(1,24,Geno, fontsize=16);
            plt.ylim(0,28)
            SubPos +=1
    Axis.spines['right'].set_visible(False)
    Axis.spines['top'].set_visible(False)
    plt.tick_params(axis='x',direction='in',labelsize = 12,rotation=30)
    try:
        CL = FLNData.loc[('L','C',cul),'Value'].median()
        CS = FLNData.loc[('S','C',cul),'Value'].median()
        WS = FLNData.loc[('S','W',cul),'Value'].median() 
        WL = FLNData.loc[('L','W',cul),'Value'].median()
        PlotLNlines(CL,CS,WS,WL,8)
        #plt.plot([18,24],[MinLN + VrnLN]*2,'-',color='k',lw=2)
        #plt.text(21,22,str(VxPLN))
    except:
        do = 'nothing'    
    
    if Pos in list(range(1,101,5)):
        plt.tick_params(axis='y', which='both', left=True,right=False, labelleft=True)
        plt.ylabel('FinalLeafNumber',fontsize=16)
    else:
        plt.tick_params(axis='y', which='both', left=True,right=False, labelleft=False)


# +
# Fig = plt.figure(figsize=(8, 80))
# Pos = 1
# for cul in Cultivars:
#     Axis = Fig.add_subplot(50,2,Pos)
#     try:
#         CL = FLNData.loc[('L','C',cul),'Value'].median()
#         CS = FLNData.loc[('S','C',cul),'Value'].median()
#         WS = FLNData.loc[('S','W',cul),'Value'].median() 
#         WL = FLNData.loc[('L','W',cul),'Value'].median()
#         camp.plotFLNs(CL,CS,WS,WL,Axis,8,20)
#         camp.plotLNlines(CL,CS,WS,WL,8)
#     except:
#         do = 'nothing'
        
#     colors = ['b','b','r','r']
#     fills = ['b','w','w','r']
#     SubPos = 0
#     for Photo,VegT in [('L','C'),('S','C'),('S','W'),('L','W')]:
#             try:
#                 y = FLNData.loc[(Photo,VegT,cul),'Value']
#             except:
#                 y = []
#             try:
#                 x = np.add(list(range(len(y))),SubPos)
#             except:
#                 x = []
#             plt.plot(x,y, 'o', markersize = 6, mec = colors[SubPos],
#                      mfc=fills[SubPos], label = Photo + " " + VegT)
#             SubPos +=1
#     plt.text(0.05,0.85,cul,transform=Axis.transAxes)
#     Pos += 1
# Fig.patch.set_facecolor('white')
# -

GenoParams = pd.DataFrame(index = Cultivars)
FLN = FLNData.select_dtypes(np.number).groupby(['Photoperiod','VegPhaseTemp','Genotype']).mean()
GenoParams.loc[:,'MinLeafNumber'] = FLN.loc[('L','C'),'Value']
GenoParams.loc[:,'FLNPPSens'] = FLN.loc[('S','C'),'Value'] - FLN.loc[('L','C'),'Value']
GenoParams.loc[:,'FLNVnSens'] = FLN.loc[('L','W'),'Value'] - FLN.loc[('L','C'),'Value']
GenoParams.loc[:,'FLN_CL'] = FLN.loc[('L','C'),'Value']
GenoParams.loc[:,'FLN_WL'] = FLN.loc[('L','W'),'Value']
GenoParams.loc[:,'FLN_CS'] = FLN.loc[('S','C'),'Value']
GenoParams.loc[:,'FLN_WS'] = FLN.loc[('S','W'),'Value']
FLN.loc[:,'HSTS'] = (FLN.loc[:,'Value'] - 2.8)/1.1
def VITS_geno(ind):
    if ind[0] == 'S':
        PPS = GenoParams.loc[ind[2],'FLNPPSens']
        return 2 + PPS/1.1
    if ind[0] == 'L':
        return 2
FLN.loc[:,'VITS'] = [VITS_geno(x) for x in FLN.index]
FLN.loc[:,'HSVI'] = FLN.loc[:,'HSTS'] - FLN.loc[:,'VITS']
FLN.sort_index(inplace=True)

NameMap = pd.read_excel('./RawData/WheatNameMapping.xlsx','NameMap',index_col=1,engine='openpyxl')

# ## Make parameter file with CAMP parameters

CampInputs = GenoParams.loc[:,['FLN_CL','FLN_WL','FLN_CS','FLN_WS']]
CampInputs.index.name = 'Cultivar'
ParamNameMap = {
    'FLN_CL': '[Phenology].CAMP.FLNparams.CL',
    'FLN_WL': '[Phenology].CAMP.FLNparams.WL', 
    'FLN_CS': '[Phenology].CAMP.FLNparams.CS',
    'FLN_WS': '[Phenology].CAMP.FLNparams.WS'
}  
CampInputs.columns = [ParamNameMap[x] for x in CampInputs.columns]
CampInputs.index = [NameMap.loc[x,'StandardName'] for x in CampInputs.index]
CampInputs.index.name = 'Cultivar'
CampInputs.loc['Whistler','[Phenology].CAMP.FLNparams.WS'] = 16.5 #  Assume SN = LN + PPS
CampInputs.loc[:,'VrnTreatTemp'] = 5.8
CampInputs.loc[:,'VrnTreatDuration'] = 60
CampInputs.loc[:,'Expt'] = 'LaTrobeCE'
CampInputs.to_excel('./ProcessedData/CEParamsFLN_Aus.xlsx',sheet_name='ObservedParams')

AllCampInputs = pd.read_excel('..\CampVrnParamInputs.xlsx',index_col=0)

graph = plt.figure(figsize=(20,15))
pan=1
locMap = {'Axe':1, 'Bolac':2,'Wyalkatchem':3}
for c in ['Axe','Bolac','Wyalkatchem']:
    params = camp.deriveVrnParams(AllCampInputs.loc[c,:])
    ax = graph.add_subplot(4,3,locMap[c])
    camp.plotVITS(params, 3,ax,c,xmax=11)
    ax = graph.add_subplot(4,3,locMap[c]+3)
    camp.boundPlots(params, ax, c,xmax=11)
    ax = graph.add_subplot(4,3,locMap[c]+6)
    camp.Vrn2Plots(params,ax,c, ymax=2.5,xmax=6)  
    ax = graph.add_subplot(4,3,locMap[c]+9)
    camp.Vrn1Plots(params, ax,c,ymax=2.5,xmax=4)  


def openAxies():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=True,top=False, labelbottom=True)
    plt.tick_params(axis='y', which='both', left=True,right=False, labelleft=True)


camp.deriveVrnParams(AllCampInputs.loc['Axe',:])

Fig = plt.figure(figsize=(10,15))
locMap = {'AXE':1, 'BOLAC':2,'WYALKATCHEM':3}
for c in ['AXE','BOLAC','WYALKATCHEM']:
    CL = FLNData.loc[('L','C',c),'Value'].median()
    CS = FLNData.loc[('S','C',c),'Value'].median()
    WS = FLNData.loc[('S','W',c),'Value'].median() 
    WL = FLNData.loc[('L','W',c),'Value'].median()
    ax = Fig.add_subplot(5,3,locMap[c])
    camp.plotFLNs(CL, CS, WS, WL,ax,8,16)
    camp.plotLNlines(CL, CS, WS, WL,8)
    plt.text(0.98,0.98,NameMap.loc[c,'StandardName'],transform=ax.transAxes, verticalalignment='bottom',horizontalalignment='right')
    params = camp.deriveVrnParams(AllCampInputs.loc[NameMap.loc[c,'StandardName'],:])
    ax = Fig.add_subplot(5,3,locMap[c]+3)
    figY_X = 1.0474365598206037
    camp.plotVITS(params, 3,ax,NameMap.loc[c,'StandardName'],xmax=11)
    openAxies()
    ax = Fig.add_subplot(5,3,locMap[c]+6)
    camp.boundPlots(params, ax, NameMap.loc[c,'StandardName'],xmax=11)
    openAxies()
    ax = Fig.add_subplot(5,3,locMap[c]+9)
    camp.Vrn2Plots(params,ax,NameMap.loc[c,'StandardName'], ymax=2.5,xmax=6)  
    openAxies()
    ax = Fig.add_subplot(5,3,locMap[c]+12)
    camp.Vrn1Plots(params, ax,NameMap.loc[c,'StandardName'],ymax=2,xmax=4,figY_X=figY_X)  
    openAxies()
plt.tight_layout()
Fig.savefig('sup4.jpg',format='jpg',dpi=300,bbox_inches="tight")     
Fig.patch.set_facecolor('white')

## Note the figY_X parameter for the script above is approximated by the height/width retrived below.  
#However the plt.tight_layout() method alters the dimensions of each plot so the graphs must be generated first, then figY_X calculated
#the the plots regenerated to give the correct text orrientation
bbox = ax.get_window_extent().transformed(Fig.dpi_scale_trans.inverted())
width, height = bbox.width, bbox.height
height/width
