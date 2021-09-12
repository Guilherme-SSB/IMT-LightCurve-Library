from os import system
system('cls')

#%%
###############################
# 
# Uncertains()
# 
# https://docs.bokeh.org/en/latest/docs/gallery/histogram.html
# https://stats.stackexchange.com/questions/154133/how-to-get-the-derivative-of-a-normal-distribution-w-r-t-its-parameters
###############################
# 
import pandas as pd
import numpy as np
# from scipy.stats import norm
# import scipy.special

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

def determine_uncertains(parameter, tolerance):
    table = pd.read_csv('final_table.csv')

    min_error = table['chi2'].min()

    data = []

    for i in range(len(table['chi2'])):
        if table['chi2'].loc[i] < min_error+tolerance:
            data.append(table[parameter].loc[i])
    return data

def stats_for_histogram(data, bins):
    return np.histogram(data, density=True, bins=bins)

def plot_histogram(data=None, bins=30):

    hist, edges = stats_for_histogram(data, bins)

    p = figure(title='Histogram plot',
          plot_width=650, plot_height=400,
          background_fill_color='#fafafa')

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color='navy', line_color='white', alpha=0.5)

    p.y_range.start = 0

    show(p)

def plot_gaussian(data, mu, sigma):
    x = np.linspace(min(data), max(data), len(data))

    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    p = figure(title='Gaussian distribuition',
          plot_width=650, plot_height=400,
          background_fill_color='#fafafa',
          x_range=(min(data), max(data)))

    p.line(x, pdf, line_color='#ff8888', line_width=4, alpha=0.7, legend_label='PDF')
    p.title.text = f'Normal Distribution Approximation (μ={round(mu,4)}, σ={round(sigma,4)})'
    p.y_range.start = 0
    p.legend.location = "center_right"
    # p.xaxis.axis_label = 'x'
    # p.yaxis.axis_label = 'Pr(x)'

    show(p)

def plot_histogram_gaussian(data, bins, mu, sigma, factor=0.005):
    hist, edges = stats_for_histogram(data, bins)

    x = np.linspace(min(data)-factor, max(data)+factor, len(data))

    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    p = figure(plot_width=650, plot_height=400,
          background_fill_color='#fafafa',
          x_range=(min(data)-factor, max(data)+factor) )

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color='navy', line_color='white', alpha=0.5)

    p.line(x, pdf, line_color='#ff8888', line_width=4, alpha=0.7, legend_label='PDF')

    p.title.text = f'Normal Distribution Approximation (μ={round(mu,12)}, σ={round(sigma,12)})'
    p.y_range.start = 0
    p.legend.location = "center_right"

    show(p)


# ['b_impact', 'p', 'period', 'adivR', 'chi2']

data = determine_uncertains('p', 1)

mu = np.mean(data) + 0.004835 
sigma = 0.0002666

# plot_histogram(data, bins=int(pd.Series(data).nunique())+1)
# plot_gaussian(data=data, mu=mu, sigma=sigma)
plot_histogram_gaussian(data=data, bins=30, mu=mu, sigma=sigma, factor=0.005)

# data = determine_uncertains('p', 1)

# hist, edges = np.histogram(data, density=True, bins=int(pd.Series(data).nunique()))

#%%
################################
## 
## .export_filters_to_csv()
##
################################
# https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.to_periodogram.html
# https://github.com/lightkurve/lightkurve/tree/main/src/lightkurve

# from imt_lightcurve.models.lightcurve import LightCurve

# import pandas as pd

# LIGHTCURVE = 'RESAMPLED_0110839339_20081116T190224'
# data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
# time = data.DATE.to_numpy()
# flux = data.WHITEFLUX.to_numpy()

# curve = LightCurve(time=time, flux=flux)
# WHERE_TO_SAVE = 'C:/Users/guisa/Desktop/filters_dataset'
# DATASET_PATH = 'C:/Users/guisa/Google Drive/01 - Iniciação Científica/IC-CoRoT_Kepler/resampled_files'

# curve.export_filters_to_csv(WHERE_TO_SAVE, DATASET_PATH, 'ideal', cutoff_freq_range=(0.1, 0.9, 0.1))
# curve.export_filters_to_csv(WHERE_TO_SAVE, DATASET_PATH, 'gaussian', cutoff_freq_range=(0.1, 0.9, 0.1))
# curve.export_filters_to_csv(WHERE_TO_SAVE, DATASET_PATH, 'butterworth', cutoff_freq_range=(0.1, 0.9, 0.1), order_range=(1, 6, 1))
# curve.export_filters_to_csv(WHERE_TO_SAVE, DATASET_PATH, 'bessel', cutoff_freq_range=(0.1, 0.9, 0.1), order_range=(1, 6, 1))
# curve.export_filters_to_csv(WHERE_TO_SAVE, DATASET_PATH, 'median', numNei_range=(3, 11, 2))


# %%
