#%%
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.visualization.data_viz import line_plot, multi_line_plot

import pandas as pd
import numpy as np
import scipy.signal as ssg


# Chosen lightcurve
LIGHTCURVE = 'RESAMPLED_0102890318_20070206T133547'


# Importing lightcurve data from github
data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()

# Create the LightCurve object
curve = LightCurve(time=time, flux=flux)
time_sampling = (pd.Series(curve.time).diff().min())*86400
frequency_sampling = 1/time_sampling


X, Y_raw = ssg.periodogram(curve.flux, frequency_sampling, detrend='constant', scaling='spectrum')
X, Y = ssg.periodogram(curve.flux, frequency_sampling, detrend='linear', scaling='spectrum')

multi_line_plot(X, Y_raw, Y, label_y1='Raw LC', label_y2='Detrend LC' , x_axis_type='log', x_range=(10**-7, 10**-3))


# # %%

# import pickle

# file_name = 'C:/Users/guisa/Desktop/freq_data.pkl'
# with open(file_name, 'rb') as file:
#     freq_data = pickle.load(file)
# # freq_data.keys()

# marcelo = freq_data['features']['spec'][5]

# print('marcelo :')
# print(np.mean(marcelo))
# print(np.min(marcelo))
# print(np.max(marcelo))

# print('\nmeu :')
# print(np.mean(Y))
# print(np.min(Y))
# print(np.max(Y))


# %%
