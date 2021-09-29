# %%
# Reading Deleuil table
import numpy as np
from imt_lightcurve.simulation.simulation import Simulate
from imt_lightcurve.models.lightcurve import LightCurve
import tabula
import pandas as pd

path = r"C:\Users\guisa\Google Drive\01 - Iniciação Científica\01 - Referências\Deleuil et. al., 2018.pdf"

lista_table_5 = tabula.read_pdf(path, pages='20', multiple_tables=True)
tabela_5 = pd.DataFrame(lista_table_5[0])
tabela_5 = tabela_5.drop(0, axis=0)


def get_true_value(corot_id: int, parameter: str):
    return float(tabela_5[tabela_5['CoRoT-ID'] == corot_id][parameter].values[0].split('±')[0])


def define_interval_period(period: float):
    return np.arange(round(period, 2)-0.02, round(period, 2)+0.03, 0.01)


def define_interval_p(p: float):
    return np.arange(round(p, 2)-0.02, round(p, 2)+0.03, 0.01)


def define_interval_adivR(adivR: float):
    return np.arange(round(adivR, 2)-0.02, round(adivR, 2)+0.03, 0.01)


def define_interval_b(b: float):
    return np.arange(round(b, 2)-0.02, round(b, 2)+0.03, 0.01)


# %%

# Chosen lightcurve
LIGHTCURVE = 'RESAMPLED_0101086161_20070516T060226'
curve_id = int(LIGHTCURVE.split('_')[1][1:])

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
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as err:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range]
               for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# window size 51, polynomial order 3
filtered_flux = savitzky_golay(windowed_curve.flux, 201, 3)
flux_error = np.std(filtered_flux)
flux_error_arr = [flux_error for i in range(len(windowed_curve.flux))]


# %%
# 1. Simulate a folded light curve
SimulationObject = Simulate()
observed_curve_lc = LightCurve(time=windowed_curve.time, flux=filtered_flux, flux_error=np.array(flux_error_arr))
x_values = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
            0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Parameters
period = get_true_value(curve_id, 'Period')
p = get_true_value(curve_id, 'Rp/R?')
adivR = get_true_value(curve_id, 'a/R?')
b = get_true_value(curve_id, 'b')


# b_values = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
b_values = define_interval_b(b)
# print('b_impact =', round(b, 2))
# print(b_values, end='\n\n')


# p_values = [0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.090]
p_values = define_interval_p(p)
# print('p =', round(p, 2))
# print(p_values, end='\n\n')


# period_values = [6, 6.1, 6.2, 6.2, 6.4 ]
period_values = define_interval_period(period)
# print('period =', round(period, 2))
# print(period_values, end='\n\n')



# adivR_values = [11.9, 11.91, 11.92, 11.93, 11.94, 11.95]
adivR_values = define_interval_adivR(adivR)
# print('avivR =', round(adivR, 2))
# print(adivR_values, end='\n\n')

final_results = SimulationObject.simulate_values(CoRoT_ID=curve_id, observed_curve=observed_curve_lc, b_values=b_values, p_values=p_values, period_values=period_values, adivR_values=adivR_values, x_values=x_values, results_to_csv=False)
final_results.head()

# best_parameters, uncertanties = SimulationObject.simulate_values(observed_curve=observed_curve_lc, b_values=b_values, p_values=p_values, period_values=period_values, adivR_values=adivR_values, x_values=x_values, results_to_csv=False)

# final_results = pd.DataFrame(dict(CoRoT_ID=[], b_impact=[], b_impact_uncertanties=[], p=[], p_uncertanties=[], period=[], period_uncertanties=[], adivR=[], adivR_uncertanties=[], chi2=[]), dtype=float)

# final_results = final_results.append(
#     dict(
#         CoRoT_ID=curve_id,
#         b_impact=best_parameters.iloc[0],
#         b_impact_uncertanties=uncertanties.iloc[0],
#         p=best_parameters.iloc[1],
#         p_uncertanties=uncertanties.iloc[1],
#         period=best_parameters.iloc[2],
#         period_uncertanties=uncertanties.iloc[2],
#         adivR=best_parameters.iloc[3],
#         adivR_uncertanties=uncertanties.iloc[3],
#         chi2=best_parameters.iloc[4]
#     ),
#     ignore_index=True)


# final_results.set_index('CoRoT_ID', inplace=True)
# final_results.head()


# %%
# from imt_lightcurve.models.lightcurve import LightCurve, PhaseFoldedLightCurve
# from imt_lightcurve.simulation.simulation import Simulate

# import pandas as pd
# import numpy as np

# # Chosen lightcurve
# LIGHTCURVE = 'RESAMPLED_0101086161_20070516T060226'

# # Importing lightcurve data from github
# data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
# time = data.DATE.to_numpy()
# flux = data.WHITEFLUX.to_numpy()

# # Create the LightCurve object
# curve = LightCurve(time=time, flux=flux)

# # Create a folded lightcurve
# folded_curve = curve.fold(window=0.15, window_filter=201, order_filter=3)
# folded_curve.plot()


# %%
