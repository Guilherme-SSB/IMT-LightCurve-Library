#%% Imports
import pandas as pd
import numpy as np

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

### Importing table
final_table = pd.read_csv('final_table.csv')

### Selecting the parameter
parameter = 'p'
parameter_table = final_table[parameter]

### Determinating the tolerance
tolerance = 0.1

min_error = final_table['chi2'].min()
data = []
chi2_data = []

for i in range(len(final_table['chi2'])):
    if final_table['chi2'].loc[i] < min_error + tolerance:
        data.append(parameter_table.loc[i])
        chi2_data.append(final_table['chi2'].loc[i])

# print(final_table)
# print(pd.Series(data).value_counts())

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
# from math import sqrt

# x_bar = np.mean(data)
# sigma = np.std(data)
# n = len(data)
# z = 1.96

# print('Confidence Interval =', round(x_bar, 4), '±', round((z*sigma)/sqrt(n),8) )


# %%
from scipy.stats import norm 

# value_counts = pd.Series(data).value_counts()
# data_fixed = np.full((1, value_counts.to_numpy()[0]), value_counts.index[0], dtype='float').ravel()

### Computing mu and sigma
mu, sigma = norm.fit(data)

# plot_histogram(data, bins=int(pd.Series(data).nunique())+1)
plot_gaussian(data, amplitude=1, mu=mu, sigma=sigma, bins=int(pd.Series(data).nunique())+1)
print('Uncertains =', round(sigma, 8))



# %%
# Libs
import matplotlib.pyplot as plt

# 1. Determinar x e y
#   x -> valores do parametro
#   y -> valores de chi2
x_data = data
y_data = chi2_data

# 2. Calcular x e y de interesse
#   tolerancia = 0.1

# 3. Scatter plot(x, y)
plt.scatter(x_data, y_data)

# 4. Centro de massa
x_mean = np.mean(x_data)
y_mean = np.mean(y_data)

plt.plot(x_mean, y_mean, marker='x', markersize=10, label='Centro de massa')
plt.title('Scatter plot dos dados originais')
plt.legend(loc=1, framealpha=1, fontsize=8)
plt.show()

# 5. Normalizando dados
x_normalized = x_data - x_mean
y_normalized = y_data - y_mean

plt.scatter(x_normalized, y_normalized)
plt.title('Scatter plot dos dados normalizados')
plt.show()

# 6. Matriz de covariância
cov_matrix = np.cov(x_normalized, y_normalized)
print('Matriz de covariância:')
print(cov_matrix, end='\n')

det_cov_matrix = np.linalg.det(cov_matrix)
print('Determinante da matriz de covariância:')
print(det_cov_matrix)

# 7. Understanding degenerate multivariate normal distribution
from scipy.stats import multivariate_normal

y = multivariate_normal.pdf(x_normalized)

plt.plot(x_normalized, y)
plt.show()



# %%
