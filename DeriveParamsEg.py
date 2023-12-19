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

cult = 'Amarok'

CampVrnParams = pd.read_excel('CampVrnParams.xlsx',index_col='Cultivar')
AmarokParams = CampVrnParams.loc[cult,:]

camp.plotVITS(AmarokParams, 3, cult)

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.boundPlots(AmarokParams, ax, cult)

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.Vrn2Plots(AmarokParams,ax,cult)     

graph  = plt.figure(figsize=(5,5))
ax = graph.add_subplot(1,1,1)
camp.Vrn1Plots(AmarokParams, ax,cult,5,3)    
