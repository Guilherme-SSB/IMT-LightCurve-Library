from imt_lightcurve.models.lightcurve import LightCurve

import pandas as pd
import numpy as np


LIGHTCURVE = 'RESAMPLED_0102912369_20070203T130553'

data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')

time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()
curve = LightCurve(time, flux)

def test_print():
    assert str(curve) == 'LightCurve Object'

def test_butterworth_two_zerothree():
    assert np.array_equal(curve.butterworth_lowpass_filter(2, 0.3).filtered_flux[0:3], np.array([189206.67949575296, 189248.5789335832, 189320.11736140097]))
