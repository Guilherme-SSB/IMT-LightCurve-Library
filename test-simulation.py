#%%
from imt_lightcurve.models.lightcurve import LightCurve
from imt_lightcurve.simulation.simulation import Simulate

import pandas as pd
import numpy as np

# Chosen lightcurve
LIGHTCURVE = 'RESAMPLED_0101086161_20070516T060226'

# Importing lightcurve data from github
data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()

normalized_flux = flux / np.median(flux)

# Create the LightCurve object
curve = LightCurve(time=time, flux=normalized_flux)

# Create a folded lightcurve
folded_curve = curve.fold()

# Windowing signal
windowed = 0.15

time = folded_curve.time
flux = folded_curve.flux

time_w = time[(time > -1*windowed) & (time < windowed)]
flux_w = flux[(time > -1*windowed) & (time < windowed)]

windowed_curve = LightCurve(time_w, flux_w)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as err:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


filtered_flux = savitzky_golay(windowed_curve.flux, 201, 3) # window size 51, polynomial order 3
flux_error = np.std(filtered_flux)
flux_error_arr = [flux_error for i in range(len(windowed_curve.flux))]

# 1. Simulate a folded light curve
SimulationObject = Simulate()
observed_curve_lc = LightCurve(time=windowed_curve.time, flux=filtered_flux, flux_error=np.array(flux_error_arr))
x_values = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0 , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 , 1.1, 1.2, 1.3, 1.4, 1.5]
# Transit impact parameter
b_values = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]

# Radius values of the planet compared to the star
p_values = [0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.090]

# Orbital period values to be considered
period_values = [6, 6.1, 6.2, 6.2, 6.4 ]

# Orbital radius values compared to star radius
adivR_values = [11.9, 11.91, 11.92, 11.93, 11.94, 11.95]

final_table_sorted_by_chi2 = SimulationObject.simulate_values(observed_curve=observed_curve_lc, b_values=b_values, p_values=p_values, period_values=period_values, adivR_values=adivR_values, x_values=x_values, results_to_csv=False)



# %%
