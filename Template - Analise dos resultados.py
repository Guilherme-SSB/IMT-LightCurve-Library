#%% Loading table
import pandas as pd
import numpy as np

from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.visualization.data_viz import multi_line_plot

results_final = pd.read_csv('results_table/FINAL_TABLE_IC.csv', index_col='CoRoT_ID')
results_final['avg_error'] = (results_final['e_period'] + results_final['e_p'] + results_final['e_adivR'] + results_final['e_b']) / 4
results_final.head()

#%%

# results_final.index.value_counts()
"""
100725706
101086161 
315239728 
315211361 
315198039
110839339 
106017681 
105891283 
105833549 
105819653
105793995 
105209106 
102912369 
102890318 
102764809
102671819 
101368192
652180991
"""
ID_ANALYSIS = 315211361

results_final.loc[ID_ANALYSIS].sort_values(by=['chi2', 'avg_error']).head()

# %%

# Importing lightcurve data from github
data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + str(ID_ANALYSIS) + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()

# Create the LightCurve object
curve = LightCurve(time=time, flux=flux)
# curve.plot()

# Filtered curve
filtered_curve = curve.median_filter(9)
filtered_curve.view_filtering_results()
# %%

folded_curve = curve.fold(str(ID_ANALYSIS))
aux = LightCurve(filtered_curve.time, filtered_curve.filtered_flux)
filtered_folded_curve = aux.fold(str(ID_ANALYSIS))

multi_line_plot(
    folded_curve.time,
    folded_curve.flux,
    filtered_folded_curve.flux,
    label_y1='Original LC',
    label_y2='Filtered LC',
    title='Folded LC Original x Filtered ',
    y_axis='Normalized flux'
)

# %%
