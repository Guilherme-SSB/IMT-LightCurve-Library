#%% Loading folded curve

import numpy as np
import pandas as pd
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import  Simulate

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
# folded_curve.plot()

################################################################
#%%
# Real values
curve_id = int(CURVE_ID)

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
from imt_lightcurve.simulation.simulation import Simulate

# FILTERS_RESULTS_PATH = 'C:/Users/guisa/Google Drive/01 - Iniciação Científica/02 - Datasets/exoplanets_confirmed/filters'
FILTERS_RESULTS_PATH = "C:/Users/guisa/Desktop/filters"
orders = [1, 2, 3, 4, 5, 6]
cutoff_freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FINAL_PATH = ''

filter_technique = 'butterworth'
final_table_results = pd.DataFrame()

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

                        ## Loading folded curve
                        path = root_dir_path + '/' + files[i]
                        curve_id = int(path.split('/')[-1].split('_')[-2][1:].strip())
                        data = pd.read_csv(path)
                        time = data.DATE.to_numpy()
                        flux = data.WHITEFLUX.to_numpy()
                        curve = LightCurve(time, flux)
                        error = np.std(curve.flux)
                        error_array = [error for i in range(len(curve.flux))]
                        curve.flux_error = error_array

                        # curve.plot(title=f'CoRoT ID: {curve_id}, {filter_technique.capitalize()} Order: {order} Cutoff Freq: {cutoff_freq}')

                        ## Extracting lightcurves real parameters
                        # Orbital period values to be considered
                        period_real = LightCurve.get_true_value(curve_id, 'Per')

                        # Radius values of the planet compared to the star
                        p_real = LightCurve.get_true_value(curve_id, 'Rp/R*')

                        # Orbital radius values compared to star radius
                        adivR_real = LightCurve.get_true_value(curve_id, 'a/R*')

                        # Transit impact parameter
                        b_real = LightCurve.get_true_value(curve_id, 'b')

                        ## Defining grid of parameters to search
                        period_values = LightCurve.define_interval_period(period_real)
                        p_values = LightCurve.define_interval_p(p_real)
                        adivR_values = LightCurve.define_interval_adivR(adivR_real)
                        b_values = LightCurve.define_interval_b(b_real)


                        ## Start the best parameters search
                        SimulationObject = Simulate()

                        try: 
                            results = SimulationObject.simulate_values(
                                CoRoT_ID=curve_id, observed_curve=curve,
                                b_values=b_values,
                                p_values=p_values,
                                period_values=period_values,
                                adivR_values=adivR_values,
                                set_best_values=True,
                                results_to_csv=False,
                                filter_technique=f'{filter_technique}',
                                filter_order=f'{order}',
                                filter_cutoff=f'{cutoff_freq}'
                            )

                            final_table_results = final_table_results.append(results)
                        
                        except:
                            # Se der problema, eu posso salvar os resultados e ja era
                            final_table_results.head()
                            raise ValueError()
                            
                        


                    final_table_results.head()
                break
            break
        break

                # for j in range(0, len(files)):
                #     if files[j].endswith('.csv'):
                #         print(files[j])

                #     break 

            





# %%
import pandas as pd
import numpy as np
from imt_lightcurve.models.lightcurve import LightCurve

LIGHTCURVE = 'RESAMPLED_0315211361_20100305T001525'
curve_id = int(LIGHTCURVE.split('_')[1][1:])

data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
flux = data.WHITEFLUX.to_numpy()
flux = flux / np.median(flux)

time = data.DATE.to_numpy()

curve = LightCurve(time=time, flux=flux)
curve.plot()
curve.fold(corot_id=curve_id).plot(title=f'Folded LC of CoRoT-ID: {curve_id}')

# %%
