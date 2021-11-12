#%%
import pandas as pd

original_table = pd.read_csv('ORIGINAL_PARAMETERS_TABLE.csv', index_col='CoRoT_ID')
original_table = original_table.drop(['filter_technique', 'filter_order', 'filter_cutoff', 'filter_numNei'], axis=1)

original_table_bigger_tolerance = pd.read_csv('ORIGINAL_PARAMETERS_TABLE_BIGGER_TOLERANCE.csv', index_col='CoRoT_ID')
original_table_bigger_tolerance = original_table_bigger_tolerance.drop(['filter_technique', 'filter_order', 'filter_cutoff', 'filter_numNei'], axis=1)

final_table = pd.read_csv('FINAL_TABLE.csv', index_col='CoRoT_ID')


# %%

original_table['avg_error'] = (original_table['e_period'] + original_table['e_p'] + original_table['e_adivR'] + original_table['e_b'])/4
original_table_bigger_tolerance['avg_error'] = (original_table_bigger_tolerance['e_period'] + original_table_bigger_tolerance['e_p'] + original_table_bigger_tolerance['e_adivR'] + original_table_bigger_tolerance['e_b'])/4
final_table['avg_error'] = (final_table['e_period'] + final_table['e_p'] + final_table['e_adivR'] + final_table['e_b'])/4

# %%
original_table.sort_values(by=['avg_error', 'chi2'])
original_table_bigger_tolerance.sort_values(by=['avg_error', 'chi2'])
final_table.sort_values(by=['avg_error', 'chi2']).head()
# final_table[final_table['e_p'] != 0].sort_values(by='e_p')