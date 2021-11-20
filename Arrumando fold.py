#%%
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
