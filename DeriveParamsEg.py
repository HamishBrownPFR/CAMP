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
from matplotlib import patches
import CAMP as camp
import copy
import json
from scipy.optimize import curve_fit
pd.set_option('display.max_rows',1000)
# %matplotlib inline

# ## Bring in data

#cult = 'Amarok'
cult = 'Kittyhawk'
#cult = 'Rosella'


CampVrnParams = pd.read_excel('CampVrnParams.xlsx',index_col='Cultivar')
cultParams = CampVrnParams.loc[cult,:]

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
CL, CS, WS, WL = cultParams[['FLN_CL','FLN_CS','FLN_WS','FLN_WL']]
camp.plotFLNs(CL, CS, WS, WL,ax,8,20)
camp.plotLNlines(CL, CS, WS, WL,8)
graph.savefig('FLN.jpg',format='jpg',dpi=300,bbox_inches="tight")

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.plotVITS(cultParams, 3, ax, cult)
graph.savefig('VITSHS.jpg',format='jpg',dpi=300,bbox_inches="tight")

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.boundPlots(cultParams, ax, cult)
graph.savefig('bounds.jpg',format='jpg',dpi=300,bbox_inches="tight")

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.Vrn2Plots(cultParams,ax,cult,xmax=17.7,ymax=2)   
graph.savefig('Vrn2.jpg',format='jpg',dpi=300,bbox_inches="tight")

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.Vrn1Plots(cultParams, ax,cult,5,2)
graph.savefig('Vrn1.jpg',format='jpg',dpi=300,bbox_inches="tight")
