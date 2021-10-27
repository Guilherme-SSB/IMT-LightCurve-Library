#%%
from imt_lightcurve.data_helper.helper import DATAHelper
from imt_lightcurve.models.lightcurve import LightCurve
import pandas as pd
import numpy as np

DATAHelper.compute_folded_curve(
    r'G:\Meu Drive\01 - Iniciação Científica\IC-CoRoT_Kepler\resampled_files',
    r'C:\Users\guisa\Desktop\folded_curves')

# ID = '100725706'
# curve_path = 'G:/Meu Drive/01 - Iniciação Científica/IC-CoRoT_Kepler/resampled_files/' + str(ID) + '.csv'
# data = pd.read_csv(curve_path)
# curve = LightCurve(data.DATE, data.WHITEFLUX)
# curve.plot()
# folded_curve, positions, width = curve.fold(ID)
# folded_curve.plot()


# %%
