# %%
# Importing a random lightcurve
import pandas as pd
import numpy as np

from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import Simulate

# Chosen a LightCurve to simulation process
LIGHTCURVE = 'RESAMPLED_0315239728_20100305T001525'


# Importing lightcurve data from github
data = pd.read_csv(
    'https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()
curve = LightCurve(time, flux)

# Folding lightcurve
folded_curve = curve.fold(window_filter=21, order_filter=3)
# print(folded_curve)
folded_curve.plot()


# %%
# Extracting lightcurve parameters

# Real parameters
curve_id = int(LIGHTCURVE.split('_')[1][1:])
print(f'CoRoT-ID: {curve_id} real parameters:\n')

# Transit impact parameter
b_real = LightCurve.get_true_value(curve_id, 'b')

# Radius values of the planet compared to the star
p_real = LightCurve.get_true_value(curve_id, 'Rp/R*')

# Orbital period values to be considered
period_real = LightCurve.get_true_value(curve_id, 'Per')

# Orbital radius values compared to star radius
adivR_real = LightCurve.get_true_value(curve_id, 'a/R*')


# Defining grid of parameters to search
b_values = LightCurve.define_interval_b(b_real)
print('b_impact =', round(b_real, 2))
print('b impact grid:', b_values, end='\n\n')

p_values = LightCurve.define_interval_p(p_real)
print('p =', round(p_real, 2))
print('p grid:', p_values, end='\n\n')

period_values = LightCurve.define_interval_period(period_real)
print('period =', round(period_real, 2))
print('period grid:', period_values, end='\n\n')


adivR_values = LightCurve.define_interval_adivR(adivR_real)
print('adivR =', round(adivR_real, 2))
print('adivR grid:', adivR_values, end='\n\n')


# %%
# Starting the parameters extracting
SimulationObject = Simulate()

# Creating folded lightcurve object
observed_curve_lc = folded_curve

final_results = SimulationObject.simulate_values(
    CoRoT_ID=curve_id, observed_curve=observed_curve_lc,
    b_values=b_values,
    p_values=p_values,
    period_values=period_values,
    adivR_values=adivR_values,
    set_best_values=True,
    results_to_csv=False)

final_results.head()

#%% 
# Simulating a lightcurve

simulated_curve = SimulationObject.simulate_lightcurve(
    observed_curve=observed_curve_lc, 
    b_impact=SimulationObject.b_impact_best,
    p=SimulationObject.p_best,
    period=SimulationObject.period_best,
    adivR=SimulationObject.adivR_best)

simulated_curve.view_simulation_results()


# %%

