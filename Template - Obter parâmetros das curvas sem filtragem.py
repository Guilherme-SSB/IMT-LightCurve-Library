#%%
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm

from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import Simulate


INPUT_PATH = 'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/original_folded_curves_with_error_fixed'

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
                    raw_parameters_table.to_csv('results_table/ORIGINAL_PARAMETERS_ERROR_FIXED.csv', index=False)
                    raise Exception('Something went wrong! Saving results and closing the script')

raw_parameters_table = raw_parameters_table.drop(['filter_technique', 'filter_order', 'filter_cutoff', 'filter_numNei'], axis=1)
raw_parameters_table.to_csv('results_table/ORIGINAL_PARAMETERS_ERROR_FIXED.csv', index=True)

# %%

