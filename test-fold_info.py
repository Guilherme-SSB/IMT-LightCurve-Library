#%%

ID = '101368192'


import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.0f}'.format})

table = pd.read_excel(r'C:\Users\guisa\Desktop\Posicao e largura dos eclipses.xlsx', index_col='CoRoT ID')
# column_name = ''
values = table['RESAMPLED_0' + ID + '_20070516T060226'].values
values = values[~np.isnan(values)]
print(repr(values))


# %%
