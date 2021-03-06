#%%
from imt_lightcurve.data_helper.helper import DATAHelper

DATAHelper.compute_folded_curve(
    r'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/filters',
    r'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/filtered_folded_curves_with_error_fixed')

# DATAHelper.compute_folded_curve_new(
#     r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\02 - Datasets\exoplanets_confirmed\csv_files',
#     r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\02 - Datasets\exoplanets_confirmed\original_folded_curves_with_error_fixed')



# %%
import numpy as np
import pandas as pd
from imt_lightcurve.models.lightcurve import LightCurve

# Chosen a LightCurve to simulation process
CURVE_ID = '100725706'

# Importing lightcurve data from github
data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + CURVE_ID + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()

# Create the LightCurve object
curve = LightCurve(time=time, flux=flux)
curve.plot()

# Folded curve
folded_curve = curve.fold(CURVE_ID)
folded_curve.plot()

# %%
