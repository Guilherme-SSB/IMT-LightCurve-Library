#%%
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm

from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import Simulate
from imt_lightcurve.visualization.data_viz import multi_line_plot


INPUT_PATH = 'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/original_folded_curves'

total_files = 0
for root_dir_path, sub_dirs, files in os.walk(INPUT_PATH):
    for file in files:
        if file.endswith('.csv'):
            total_files += 1
    pass
print(f'There are {total_files} curves to test\n\n')
# %%
raw_parameters_table = pd.DataFrame()

print('Starting simulation...')
with tqdm(range(total_files), colour='blue', desc='Simulating') as pbar:
    for root_dir_path, sub_dirs, files in os.walk(INPUT_PATH):
        for file in files:
            if file.endswith('.csv'):
                CURVE_PATH = os.path.join(root_dir_path, file)
                CURVE_PATH = CURVE_PATH.replace("\\", "/")
                curve_id = CURVE_PATH.split('/')[-1].split('_')[-1].split('.')[0]

                data = pd.read_csv(CURVE_PATH)
                curve = LightCurve(time=data.TIME, flux=data.FOLD_FLUX, flux_error=data.ERROR)
                curve_id_int = int(curve_id)
                ## Orbital period values to be considered
                period_real = LightCurve.get_true_value(curve_id_int, 'Per')

                ## Radius values of the planet compared to the star
                p_real = LightCurve.get_true_value(curve_id_int, 'Rp/R*')

                ## Orbital radius values compared to star radius
                adivR_real = LightCurve.get_true_value(curve_id_int, 'a/R*')

                ## Transit impact parameter
                b_real = LightCurve.get_true_value(curve_id_int, 'b')

                # Defining grid of parameters to search
                period_values = LightCurve.define_interval_period(period_real)
                p_values = LightCurve.define_interval_p(p_real)
                adivR_values = LightCurve.define_interval_adivR(adivR_real)
                b_values = LightCurve.define_interval_b(b_real)

                # Simulating grid
                try:
                    SimulationObject = Simulate()

                    results = SimulationObject.simulate_values(
                        CoRoT_ID=curve_id_int,
                        observed_curve=curve,
                        b_values=b_values,
                        p_values=p_values,
                        period_values=period_values,
                        adivR_values=adivR_values,
                        set_best_values=True,
                        results_to_csv=False,
                        filter_technique="",
                        filter_order="",
                        filter_cutoff="",
                        filter_numNei="")
                        
                    raw_parameters_table = raw_parameters_table.append(results)
                    pbar.update(1)
                except:
                    raw_parameters_table.to_csv('ORIGINAL_PARAMETERS.csv', index=False)
                    raise Exception('Something went wrong! Saving results and closing the script')

raw_parameters_table = raw_parameters_table.drop(['filter_technique', 'filter_order', 'filter_cutoff', 'filter_numNei'], axis=1)
raw_parameters_table.to_csv('ORIGINAL_PARAMETERS_TABLE.csv', index=True)

# %%


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
ID_ANALYSIS = 105891283

raw_parameters_table.loc[ID_ANALYSIS]#.sort_values(by='chi2').head()

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
