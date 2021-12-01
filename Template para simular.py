#%% Loading folded curve

import numpy as np
import pandas as pd
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import  Simulate

# Chosen a LightCurve to simulation process
CURVE_ID = '315198039'

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

real_parameters = [period_real, adivR_real, p_real, b_real]
print(real_parameters)

# array([4.0377, 4.0378, 4.0379, 4.038 , 4.0381, 4.0382])

# Defining grid of parameters to search
period_values = LightCurve.define_interval_period(period_real)
adivR_values = LightCurve.define_interval_adivR(adivR_real)
p_values = LightCurve.define_interval_p(p_real)
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
    results_to_csv=False,
    filter_technique='TESTE',
    filter_order='TESTE',
    filter_cutoff='TESTE',
    filter_numNei='TESTE')

final_results = final_results.drop(['filter_technique', 'filter_order', 'filter_cutoff', 'filter_numNei'], axis=1)
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

