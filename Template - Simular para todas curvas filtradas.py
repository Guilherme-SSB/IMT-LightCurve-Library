#%% Imports
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import Simulate

INPUT_PATH = 'C:/Users/guisa/Desktop/Arquivos_IC/folded_curves'

total_files = 0
for root_dir_path, sub_dirs, files in os.walk(INPUT_PATH):
    for file in files:
        total_files += 1
    pass
print(f'There are {total_files} curves to test\n\n')

# %%
final_table = pd.DataFrame()

print('Starting simulation...')
with tqdm(range(total_files), colour='blue', desc='Simulating') as pbar:
    for root_dir_path, sub_dirs, files in os.walk(INPUT_PATH):
        for file in files:
            n = ""
            f = ""
            numNei = ""
            CURVE_PATH = os.path.join(root_dir_path, file)
            CURVE_PATH = CURVE_PATH.replace("\\", "/")
            curve_id = CURVE_PATH.split('/')[-1].split('_')[-1].split('.')[0]
            filter_technique = CURVE_PATH.split('/')[6]

            if filter_technique == 'bessel':
                n = CURVE_PATH.split('/')[-3]
                f = CURVE_PATH.split('/')[-2]
                title = f'LC {curve_id}. Bessel {n} and {f}'

            if filter_technique == 'butterworth':
                n = CURVE_PATH.split('/')[-3]
                f = CURVE_PATH.split('/')[-2]
                title = f'LC {curve_id}. Butterworth {n} and {f}'

            if filter_technique == 'gaussian':
                f = CURVE_PATH.split('/')[-2]
                title = f'LC {curve_id}. Gaussian {f}'

            if filter_technique == 'ideal':
                f = CURVE_PATH.split('/')[-2]
                title = f'LC {curve_id}. Ideal {f}'

            if filter_technique == 'median':
                numNei = CURVE_PATH.split('/')[-2][-1:]
                title = f'LC {curve_id}. Median numNei {numNei}'

            # Reading a curve
            data = pd.read_csv(CURVE_PATH)
            curve = LightCurve(time=data.TIME, flux=data.FOLD_FLUX, flux_error=data.ERROR)
            # curve.plot(title=title)

            # Extracting the "real" parameters
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
                    filter_technique=filter_technique,
                    filter_order=n,
                    filter_cutoff=f,
                    filter_numNei=numNei)
                    
                final_table = final_table.append(results)
                pbar.update(1)
            except:
                final_table.to_csv('FINAL_TABLE.csv', index=False)
                raise Exception('Something went wrong! Saving results and closing the script')

                
final_table.to_csv('FINAL_TABLE.csv', index=True)



# %%

# results_final = pd.read_csv('FINAL_TABLE.csv', index_col='CoRoT_ID')

# results_final.sort_values(by='e_period').head()                                       # butterworth, n1, f01 (e_period = 0.0141421356237306)
# results_final.sort_values(by='e_p').head()                                            # butterworth, n3, f06 (e_p = 0)  
# results_final.sort_values(by='e_p')[results_final.sort_values(by='e_p')['e_p'] != 0]  # butterworth, n2, f06 (e_p = 3.469446951953614e-18) 
# results_final.sort_values(by='e_adivR').head()                                        # ideal, f09 (e_adivR = 0.0141421356237301)
# results_final.sort_values(by='e_b').head()                                            # ideal, f05 (e_b = 0)
# results_final.sort_values(by='e_b')[results_final.sort_values(by='e_b')['e_b'] != 0]  # ideal, f06 (e_b = 1.1102230246251563e-16)




# %%
