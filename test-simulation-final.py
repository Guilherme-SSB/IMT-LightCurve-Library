#%% Loading folded curve

import numpy as np
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import FILTERS_RESULTS_PATH, Simulate

# Chosen a LightCurve to simulation process
LIGHTCURVE = 'RESAMPLED_0101206560_20070516T060226'


time = np.loadtxt(r"C:\Users\guisa\Desktop\time_folded.txt")
flux = np.loadtxt(r"C:\Users\guisa\Desktop\flux_folded.txt")
folded_curve = LightCurve(time, flux)
error = np.std(folded_curve.flux)
error_array = [error for i in range(len(folded_curve.flux))]
folded_curve.flux_error = error_array

folded_curve.plot()

################################################################
# Real values
curve_id = int(LIGHTCURVE.split('_')[1][1:])

## Orbital period values to be considered
period_real = LightCurve.get_true_value(curve_id, 'Per')

## Radius values of the planet compared to the star
p_real = LightCurve.get_true_value(curve_id, 'Rp/R*')

## Orbital radius values compared to star radius
adivR_real = LightCurve.get_true_value(curve_id, 'a/R*')

## Transit impact parameter
b_real = LightCurve.get_true_value(curve_id, 'b')

real_parameters = [period_real, p_real, adivR_real, b_real]

# Defining grid of parameters to search
period_values = LightCurve.define_interval_period(period_real)
p_values = LightCurve.define_interval_p(p_real)
adivR_values = LightCurve.define_interval_adivR(adivR_real)
b_values = LightCurve.define_interval_b(b_real)


# %% Simulation
SimulationObject = Simulate()

## Simulating values
final_results = SimulationObject.simulate_values(
    CoRoT_ID=curve_id,
    observed_curve=folded_curve,
    b_values=b_values,
    p_values=p_values,
    period_values=period_values,
    adivR_values=adivR_values,
    set_best_values=True,
    results_to_csv=False)

final_results.head()

# %% Simulating a lightcurve

simulated_curve = SimulationObject.simulate_lightcurve(
    observed_curve=folded_curve, 
    b_impact=SimulationObject.b_impact_best,
    p=SimulationObject.p_best,
    period=SimulationObject.period_best,
    adivR=SimulationObject.adivR_best)

simulated_curve.view_simulation_results()


# %%
import os
import pandas as pd
import numpy as np
from imt_lightcurve.models.lightcurve import LightCurve

# FILTERS_RESULTS_PATH = 'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/filters'
FILTERS_RESULTS_PATH = "C:/Users/guisa/Desktop/filters"
orders = [1, 2, 3, 4, 5, 6]
cutoff_freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FINAL_PATH = ''

filter_technique = 'butterworth'


## Definindo paths para cada sub-pasta
if filter_technique.upper() == 'BUTTERWORTH':
    FILTERS_RESULTS_PATH += '/butterworth'
    for order in orders:
        for cutoff_freq in cutoff_freqs:
            FINAL_PATH = ''
            FINAL_PATH += FILTERS_RESULTS_PATH + f'/n{order}/f0{str(int(cutoff_freq*10))}'
            # print(FINAL_PATH, end='\n\n')
            # input()
            for root_dir_path, sub_dirs, files in os.walk(FINAL_PATH):
                for i in range(0, len(files)):
                    if files[i].endswith('.csv'):
                        path = root_dir_path + '/' + files[i]
                        curve_id = path.split('/')[-1].split('_')[-2][1:]
                        data = pd.read_csv(path)
                        time = data.DATE.to_numpy()
                        flux = data.WHITEFLUX.to_numpy()

                        curve = LightCurve(time, flux)
                        error = np.std(curve.flux)
                        error_array = [error for i in range(len(curve.flux))]
                        curve.flux_error = error_array

                        curve.plot(title=f'CoRoT ID: {curve_id}, {filter_technique.capitalize()} Order: {order} Cutoff Freq: {cutoff_freq}')

                    break
                break
            break
        break

                # for j in range(0, len(files)):
                #     if files[j].endswith('.csv'):
                #         print(files[j])

                #     break 

            





# %%
