# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sqlite3
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import MathsUtilities as MUte
import matplotlib.patheffects as path_effects
import GraphHelpers as GH
# %matplotlib inline

csfont = {'fontname':'Comic Sans MS'}

RawData = pd.read_excel('BrookingAndJamiesonRawData.xlsx',sheet_name='LeafData')
def MakeSimName(x):
    Name = 'PalmerstonNorthCETreat'
    Name += RawData.loc[x,'Temp']
    Name += RawData.loc[x,'Pp']
    Name += 'Cv'
    Name += RawData.loc[x,'Geno']
    Name += 'Durat'
    Name += str(int(RawData.loc[x,'DurationWeeks']))
    return Name
RawData.loc[:,'SimulationName'] = [MakeSimName(x) for x in RawData.index] 

RawData.set_index(['Geno','Temp','Pp','DurationWeeks','DAS'],inplace=True,drop=False)
Means = RawData.groupby(level=['Geno','Temp','Pp','DurationWeeks']).mean()
Genotypes = Means.index.get_level_values(0).drop_duplicates()
def MakeSimNameIndexedDF(x):
    Name = 'PalmerstonNorthCETreat'
    Name += x[1]
    Name += x[2]
    Name += 'Cv'
    Name += x[0]
    Name += 'Durat'
    Name += str(int(x[3]))
    return Name
Means.loc[:,'SimulationName'] = [MakeSimNameIndexedDF(x) for x in Means.index]
Means.loc[:,'Wheat.Phenology.CurrentStageName'] = 'HarvestRipe'

cols = ['g','orange']
alphas = [1,0.75,0.5,0.25]
sizes = [5,10,15,20]
style = [u'$1$', u'$5$', u'$8$', u'$11$']
Graph = plt.figure(figsize=(17,10))
ax = Graph.add_subplot(1,1,1)
z=0
for T in [('1oC','0h'),('5oC','16h'),('8oC','16h'),('11oC','16h')]:
    p = 0
    for G in Genotypes:
        plt.plot(Means.loc[(G,T[0],T[1]),'DurationDays'],
                 Means.loc[(G,T[0],T[1]),'FLN'],'o-',
                 color = cols[p],ms=30, mfc='w',mew=3, lw=2,
                 label = T[0] + ' '+ G)
        plt.plot(Means.loc[(G,T[0],T[1]),'DurationDays'],
                 Means.loc[(G,T[0],T[1]),'FLN'],marker=style[z],
                 ls='None', ms=20, color='k',
                 label = T[0] + ' '+ G)
        p += 1
    z += 1
plt.tick_params(labelsize=24)
plt.ylabel('Final Leaf Number',fontsize=32,**csfont)
plt.xlabel('Days under non-control conditions',fontsize=32,**csfont)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.text(0,10.8,'Batten Spring',fontsize=28,color=cols[0],fontweight='bold',**csfont)
plt.text(23,20,'Batten Winter',fontsize=28,color=cols[1],fontweight='bold',**csfont)
plt.text(-12,21.5,'Vernalisation Response (16 h Pp)',fontsize = 38,fontweight='bold',**csfont)

cols = ['r','y']
alphas = [1,0.75,0.5,0.25]
sizes = [5,10,15,20]
style = [u'$8$',u'$5$',  u'$11$']
Graph = plt.figure(figsize=(17,10))
ax = Graph.add_subplot(1,1,1)
z=0
G = 'Spring'
for T in ['8oC', '5oC', '11oC']:
    p = 0
    for P in ['8h','16h']:
        plt.plot(Means.loc[(G,T,P),'DurationDays'],
                 Means.loc[(G,T,P),'FLN'],'o-',
                 color = cols[p],ms=30, mfc='w',mew=3, lw=2,
                 label = T[0] + ' '+ G)
        plt.plot(Means.loc[(G,T,P),'DurationDays'],
                 Means.loc[(G,T,P),'FLN'],marker=style[z],
                 ls='None', ms=20, color='k',
                 label = T[0] + ' '+ G)
        p += 1
    z += 1
plt.tick_params(labelsize=24)
plt.ylabel('Final Leaf Number',fontsize=32,**csfont)
plt.xlabel('Days under non-control conditions',fontsize=32,**csfont)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.text(-12,13.5,'Photoperiod Response (Batten Spring)',fontsize = 38,fontweight='bold',**csfont)
plt.text(75,8.1,'16 h Pp',fontsize=28,color=cols[1],fontweight='bold',**csfont)
plt.text(60,11,'8 h Pp',fontsize=28,color=cols[0],fontweight='bold',**csfont)

cols = ['r','y']
alphas = [1,0.75,0.5,0.25]
sizes = [5,10,15,20]
style = [u'$5$', u'$8$', u'$11$',u'$25$']
Graph = plt.figure(figsize=(17,10))
ax = Graph.add_subplot(1,1,1)
G = 'Winter'
z=0
for T in ['5oC','8oC','11oC','25oC']:
    p = 0
    for P in ['8h','16h']:
        try:
            plt.plot(Means.loc[(G,T,P),'DurationDays'],
                     Means.loc[(G,T,P),'FLN'],'o-',
                     color = cols[p],ms=30, mfc='w',mew=3, lw=2,
                     label = T[0] + ' '+ G)
            plt.plot(Means.loc[(G,T,P),'DurationDays'],
                     Means.loc[(G,T,P),'FLN'],marker=style[z],
                     ls='None', ms=20, color='k',
                     label = T[0] + ' '+ G)
        except:
            dummy = 'dumb'
        p += 1
    z += 1
plt.tick_params(labelsize=24)
plt.ylabel('Final Leaf Number',fontsize=32,**csfont)
plt.xlabel('Days under non-control conditions',fontsize=32,**csfont)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.text(0,10.8,'16 h Pp',fontsize=28,color=cols[1],fontweight='bold',**csfont)
plt.text(0,15,'8 h Pp',fontsize=28,color=cols[0],fontweight='bold',**csfont)
plt.text(-12,21.5,'Vern x Pp Response (Batten Winter)',fontsize = 38,fontweight='bold',**csfont)

MeansByDate = RawData.groupby(level=['Geno','Temp','Pp','DurationWeeks','DAS']).mean()
MeansByDate.loc[:,'SimulationName'] = [MakeSimNameIndexedDF(x) for x in MeansByDate.index]
StartDate = datetime.date(2000,1,1)
MeansByDate.loc[:,'Date'] = [StartDate + datetime.timedelta(int(MeansByDate.loc[x,'DAS'])) for x in MeansByDate.index]

cols = ['r','y']
style = [u'$5$', u'$8$', u'$11$',u'$25$']
Graph = plt.figure(figsize=(10,17))
pos=1
for T in ['1oC','5oC','8oC','11oC','25oC']:
    ax = Graph.add_subplot(3,2,pos)
    for G in ['Winter','Spring']:
        for P in ['0h','8h','16h']:
            try:
                plt.plot(MeansByDate.loc[(G,T,P),'DAS'],
                         MeansByDate.loc[(G,T,P),'Haun'])
            except:
                dummy = 'dumb'
    pos +=1

cols = ['b','r','y']
fillcols = ['b','r','y','w','w','w']
style = [u'$5$', u'$8$', u'$11$',u'$25$']
Graph = plt.figure(figsize=(10,17))
pos=1
BP =  110
for T in ['1oC','5oC','8oC','11oC','25oC']:
    ax = Graph.add_subplot(3,2,pos)
    filpos = 0
    for G in ['Winter','Spring']:
        colpos = 0
        for P in ['0h','8h','16h']:
            try:
                plt.plot(MeansByDate.loc[(G,T,P),'ThermalTime'],
                         MeansByDate.loc[(G,T,P),'Haun'],'o',
                        mec=cols[colpos],mfc=fillcols[filpos])
                plt.plot([BP*1,
                          BP*1+BP*0.75*2,
                          BP*1+BP*0.75*2+BP*6,
                          BP*1+BP*0.75*2+BP*6+11*BP*1.4],
                          [0,2,8,18],'-',color='k')
            except:
                dummy = 'dumb'
            colpos +=1
            filpos +=1
            plt.text(0.05,0.95,T,transform=ax.transAxes)
    pos +=1

HaunDF = MeansByDate.reindex(['SimulationName','Date','Haun'],axis=1)
HaunDF.columns = ['SimulationName','Date','Wheat.Structure.HaunStage']
FLNDF = Means.reindex(['SimulationName','Wheat.Phenology.CurrentStageName','FLN'],axis=1)
FLNDF.columns = ['SimulationName','Wheat.Phenology.CurrentStageName','Wheat.Structure.FinalLeafNumber']
FLNDF.sort_index(inplace=True)

TSFIRawData = pd.read_excel('BrookingAndJamiesonRawData.xlsx',sheet_name='TS and FI data')
TSFIRawData.set_index(['Geno','Temp','Pp','DurationWeeks'],inplace=True)
TSFIRawData.sort_index(inplace=True)
#TSFIRawData.loc[:,'SimulationName'] = [MakeSimNameIndexedDF(x) for x in TSFIRawData.index]

TSFIDF = TSFIRawData.reindex(['FLN','EJ_haunStage(fitted)','TS_HaunStage(fitted)'],axis=1)
TSFIDF.columns= ['FLN', 'HSFI', 'HSTS']

FLNDF = pd.concat([FLNDF,TSFIDF],axis=1,sort=True)

## check FLN from the two sources is the same
plt.plot(FLNDF.loc[:,'FLN'],FLNDF.loc[:,'Wheat.Structure.FinalLeafNumber'],'o')

plt.plot(FLNDF.loc[:,'HSTS'],FLNDF.loc[:,'FLN'],'o')
plt.plot([3,13],[5.7,17.5],'-','k')

plt.plot(FLNDF.loc[:,'HSFI'],FLNDF.loc[:,'HSTS'],'o')
plt.plot([2,12],[4,14],'-','k')

HaunDF.columns = ['SimulationName', 'Clock.Today', 'Wheat.Structure.HaunStage']#.drop('FLN',axis=1)
FLNDF.drop('FLN',axis=1,inplace=True)

FLNDF.columns = ['SimulationName', 'Wheat.Phenology.CurrentStageName',
       'Wheat.Structure.FinalLeafNumber', 'Wheat.Structure.HaunStageFloralInitiation', 'Wheat.Structure.HaunStageTerminalSpikelet']

Brooking_Jamieson = pd.concat([HaunDF,FLNDF],sort=True)
Brooking_Jamieson.set_index('SimulationName',inplace=True)
Brooking_Jamieson.to_excel('C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\PalmerstonNorthCE_Obs.xlsx',sheet_name='Observed')

Brooking_Jamieson
