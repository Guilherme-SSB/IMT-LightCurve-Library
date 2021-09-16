#%% Imports
import pandas as pd
import numpy as np

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

### Importing table
final_table = pd.read_csv('final_table.csv')

### Selecting the parameter
parameter = 'b_impact'
parameter_table = final_table[parameter]

### Determinating the tolerance
tolerance = 0.05

min_error = final_table['chi2'].min()
data = []
for i in range(len(final_table['chi2'])):
    if final_table['chi2'].loc[i] < min_error + tolerance:
        data.append(parameter_table.loc[i])

print(final_table)
print(pd.Series(data).value_counts())

### Defining some help funcions

def plot_histogram(data=None, bins=30):

    hist, edges = np.histogram(data, density=True, bins=bins)

    p = figure(title='Histogram plot',
          plot_width=650, plot_height=400,
          background_fill_color='#fafafa')

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color='navy', line_color='white', alpha=0.5)

    p.y_range.start = 0

    show(p)

def plot_gaussian(data, amplitude, mu, sigma, bins, factor=0.005):
    hist, edges = np.histogram(data, density=True, bins=bins)

    x = np.linspace(min(data)-factor, max(data)+factor, len(data))

    pdf = amplitude * (1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2)))

    p = figure(plot_width=650, plot_height=400,
          background_fill_color='#fafafa',
          x_range=(min(data)-factor, max(data)+factor) )

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color='navy', line_color='white', alpha=0.5)

    p.line(x, pdf, line_color='#ff8888', line_width=4, alpha=0.7, legend_label='PDF')

    p.title.text = f'Normal Distribution Approximation (amp = {amplitude}, μ={round(mu,12)}, σ={round(sigma,12)})'
    p.y_range.start = 0
    p.legend.location = "center_right"

    show(p)


#%%
from math import sqrt

x_bar = np.mean(data)
sigma = np.std(data)
n = len(data)
z = 1.96


print('Confidence Interval =', round(x_bar, 4), '±', round((z*sigma)/sqrt(n),8) )


# %%
# from scipy.stats import norm 

# ### Computing mu and sigma
# mu, sigma = norm.fit(data)

# # plot_histogram(data, bins=int(pd.Series(data).nunique())+1)
# plot_gaussian(data, amplitude=1, mu=mu, sigma=sigma, bins=int(pd.Series(data).nunique())+1)




# %%
