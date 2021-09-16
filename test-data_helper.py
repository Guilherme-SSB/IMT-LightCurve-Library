from os import system
system('cls')

# from imt_lightcurve.data_helper.read_fits import FITSHelper

# FITSHelper.fits_to_csv(r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\02 - Datasets\eclipsing_binaries\eclipsing_binaries_fits', r'C:\Users\guisa\Desktop\eclipsing_binaries_csv')

# sample_size = FITSHelper.get_median_sample_size(r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\IC-CoRoT_Kepler\resampled_files')
# FITSHelper.resampling_dataset(r'C:\Users\guisa\Desktop\eclipsing_binaries_csv', r'C:\Users\guisa\Desktop\resampled_eclipsing_binaries_csv', sample_size)

import os 
import pandas as pd
import numpy as np
from imt_lightcurve.models.lightcurve import LightCurve

CSV_PATH = r'C:\Users\guisa\Desktop\resampled_eclipsing_binaries_csv'
# CSV_PATH = r'C:\Users\guisa\Google Drive\01 - Iniciação Científica\IC-CoRoT_Kepler\resampled_files'


tamanhos = []
for root_dir_path, sub_dirs, files in os.walk(CSV_PATH):
    for j in range(0, len(files)):
        if files[j].endswith('.csv'): 
            path = root_dir_path + "/" + files[j]
            data = pd.read_csv(path)
            time = data['DATE'].to_numpy()
            flux = data['WHITEFLUX'].to_numpy()
            curve = LightCurve(time, flux)
            tamanhos.append(len(curve.time))

print(min(tamanhos))
print(max(tamanhos))
# print(min(tamanhos))



            
