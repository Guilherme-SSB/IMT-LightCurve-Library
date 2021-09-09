from os import system
system('cls')

################################
## 
## Simulate()
##
################################

# from imt_lightcurve.simulation.simulation import Simulate
# from imt_lightcurve.models.lightcurve import LightCurve

# path = r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\IC-CoRoT_Kepler\Light Curve Simulation'

# # Planet coordinate, along the x-axis, as a function of the start's radius
# x_values_path = path+r'\files\Valores_x_simulacao.txt'
# x_values = np.loadtxt(x_values_path, dtype='float', delimiter='\n')

# # Transit impact parameter
# b_values_path =  path+r'\files\Valores_b_simulacao_rodada2.txt'
# b_values = np.loadtxt(b_values_path, dtype='float', delimiter='\n')

# # Radius values of the planet compared to the star
# p_values_path =  path+r'\files\Valores_p_simulacao_rodada2.txt'
# p_values = np.loadtxt(p_values_path, dtype='float', delimiter='\n')

# # Orbital period values to be considered
# period_values_path =  path+r'\files\Valores_periodo_simulacao_rodada2.txt'
# period_values = np.loadtxt(period_values_path, dtype='float', delimiter='\n')

# # Orbital radius values compared to star radius
# adivR_values_path =  path+r'\files\Valores_adivR_simulacao_rodada2.txt'
# adivR_values = np.loadtxt(adivR_values_path, dtype='float', delimiter='\n')

# # Observed curve
# time = [-0.1481600000,-0.1422336000,-0.1363072000,-0.1303808000,-0.1244544000,-0.1185280000,-0.1126016000,-0.1066752000,-0.1007488000,-0.0948224000,-0.0888960000,-0.0829696000,-0.0770432000,-0.0711168000,-0.0651904000,-0.0592640000,-0.0533376000,-0.0474112000,-0.0414848000,-0.0355584000,-0.0296320000,-0.0237056000,-0.0177792000,-0.0118528000,-0.0059264000,0.0000000000,0.0059264000,0.0118528000,0.0177792000,0.0237056000,0.0296320000,0.0355584000,0.0414848000,0.0474112000,0.0533376000,0.0592640000,0.0651904000,0.0711168000,0.0770432000,0.0829696000,0.0888960000,0.0948224000,0.1007488000,0.1066752000,0.1126016000,0.1185280000,0.1244544000,0.1303808000,0.1363072000,0.1422336000,0.1481600000]
# flux = [0.9998890000,0.9998620000,0.9998410000,0.9998240000,0.9998330000,0.9998400000,0.9998260000,0.9997920000,0.9997690000,0.9997670000,0.9997310000,0.9996390000,0.9994670000,0.9991300000,0.9985820000,0.9977730000,0.9966470000,0.9952250000,0.9935850000,0.9918480000,0.9901580000,0.9886300000,0.9873990000,0.9865560000,0.9860810000,0.9859430000,0.9861100000,0.9865810000,0.9873940000,0.9885630000,0.9900410000,0.9917440000,0.9935490000,0.9953280000,0.9969150000,0.9981570000,0.9990020000,0.9994800000,0.9996950000,0.9997550000,0.9997350000,0.9996680000,0.9995930000,0.9995660000,0.9996100000,0.9997080000,0.9998520000,1.0000200000,1.0001600000,1.0002200000,1.0002200000]
# flux_error = [0.0004429030,0.0004730390,0.0005336970,0.0006176250,0.0006609910,0.0006181630,0.0005694660,0.0007174970,0.0010042000,0.0012532200,0.0014065300,0.0015066500,0.0015661100,0.0015462300,0.0014475700,0.0013270600,0.0012236200,0.0011347000,0.0010342600,0.0009445330,0.0008957230,0.0009176060,0.0010341200,0.0012847100,0.0016077100,0.0018928900,0.0020054000,0.0018970600,0.0016677200,0.0014806900,0.0013842700,0.0013265800,0.0012413100,0.0011207500,0.0009860860,0.0009293850,0.0010354500,0.0011807500,0.0012602800,0.0012458800,0.0011885000,0.0011429700,0.0011507400,0.0012024600,0.0012284900,0.0011691800,0.0010192300,0.0008780320,0.0007979110,0.0007564010,0.0007313150]

# observed_curve = LightCurve(time, flux, flux_error)

# # Create a Simulate object
# SimulateObject = Simulate()
# final_table_sorted_by_chi2 = SimulateObject.simulate_values(observed_curve=observed_curve, b_values=b_values, p_values=p_values, period_values=period_values, adivR_values=adivR_values, x_values=x_values, set_best_values=True, results_to_csv=False)
# print(final_table_sorted_by_chi2.head())

# ## Build the lightcurve with the best parameters computeds
# simulated_curve = SimulateObject.simulate_lightcurve(observed_curve=observed_curve, x_values=x_values)

# ## Chi2 Error
# chi2 = simulated_curve.compare_results(see_values=False)

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
## .fold()
##
################################
# https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.to_periodogram.html
# https://github.com/lightkurve/lightkurve/tree/main/src/lightkurve

from imt_lightcurve.models.lightcurve import LightCurve

import pandas as pd

# import lightkurve as lk
# import numpy as np

LIGHTCURVE = 'RESAMPLED_0110839339_20081116T190224'
data = pd.read_csv('https://raw.githubusercontent.com/Guilherme-SSB/IC-CoRoT_Kepler/main/resampled_files/' + LIGHTCURVE + '.csv')
time = data.DATE.to_numpy()
flux = data.WHITEFLUX.to_numpy()

curve = LightCurve(time=time, flux=flux)
folded_curve = curve.fold()
folded_curve.plot()
folded_curve.median_filter(51).view_filtering_results()




# %%
