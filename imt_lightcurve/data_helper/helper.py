import os
import shutil
from statistics import median

import numpy as np
import pandas as pd
import scipy.signal as ssg
from astropy.io import fits
from tqdm import tqdm

from imt_lightcurve.models.lightcurve import LightCurve


class DATAHelper():
    # https://www.geeksforgeeks.org/class-method-vs-static-method-python/

    @staticmethod
    def fits_to_csv(FITS_PATH: str, CSV_PATH: str):
        # if csv folder do not exist, create it
        if not os.path.isdir(CSV_PATH):
            os.mkdir(CSV_PATH)

        # Verify if all files have been converted                
        path, dirs, files = next(os.walk(FITS_PATH))
        num_fits = 0
        for i in files:
            if i.endswith('.fits'):
                num_fits += 1

        for root_dir_path, sub_dirs, files in os.walk(FITS_PATH):        
            for i in range(0, len(files)):
                path = FITS_PATH + '/' + files[i]
                
                if path.endswith('.fits'):
                    # ``image_file`` contains all the information storage into .fits file
                    image_file = fits.open(path)
                    
                    # ``scidata`` contains the information of the third table of the .fits file
                    scidata = image_file[3].data

                    # ``data_aux`` contains the same info as ``scidata``, but in a np.array format
                    data_aux = np.array(scidata).byteswap().newbyteorder()

                    # ``data`` contains the same info as ``data_aux``, but in a pd.DataFrame format
                    data = pd.DataFrame(data_aux)

                    # We don't need STATUSSYS column
                    data.drop('STATUSSYS', axis=1, inplace=True)

                    # Renaming columns
                    data.rename(columns={'DATEBARTT': 'DATE'}, inplace=True)
                    data.rename(columns={'WHITEFLUXSYS': 'WHITEFLUX'}, inplace=True)

                    # Verify if there's any Not a Number (NaN) value
                    if (data.isnull().values.any()):
                        print("There's some NaN value on data. Check the data!")
                        print("Name file:", files[i])
                        break

                    # Renaming .csv files
                    name = path[path.rfind('/')+1:path.rfind('.')] + '.csv'
                    name = name.split('_')[3] + "_" + name.split('_')[4] + ".csv"

                    # Saving data
                    data.to_csv(name, index=False)
                    # print("Data saved:", name)

                    # Move to .csv folder
                    shutil.move(name, CSV_PATH)
                        

        
        path, dirs, files = next(os.walk(CSV_PATH))
        num_csv = 0
        for i in files:
            if i.endswith('.csv'):
                num_csv += 1

        if num_fits == num_csv:
            print("All files have been converted successfully!")
        else:
            print("Not all files have been converted! Uncomment the `print` statement on line 63 to see details")

    @staticmethod
    def get_median_sample_size(CSV_PATH: str) -> int:
        sample_size = []
        for root_dir_path, sub_dirs, files in os.walk(CSV_PATH):
            for j in range(0, len(files)):
                if files[j].endswith('.csv'): 
                    path = root_dir_path + "/" + files[j]
                    data = pd.read_csv(path)
                    
                    sample_size.append(len(data.WHITEFLUX))

        # Uncomment to see details
        # print("Minimum size:", min(sample_size))
        # print("Median size:", median(sample_size))
        # print("Maximum size:", max(sample_size))

        return int(median(sample_size))


    @staticmethod
    def resampling_dataset(CSV_PATH: str, RESAMPLE_PATH: str, sample_size):
        # Verify if all files have been resampled                
        path, dirs, files = next(os.walk(CSV_PATH))
        num_csv = 0
        for i in files:
            if i.endswith('.csv'):
                num_csv += 1

        count = 0
        for root_dir_path, sub_dirs, files in os.walk(CSV_PATH):
            for j in range(0, len(files)):
                if files[j].endswith('.csv'):
                    path = root_dir_path + "/" + files[j]
                    data = pd.read_csv(path)

                    # If sample size is less than 584, the lightcurve will be discarded
                    if (data.WHITEFLUX.size > 584):
                        flux = data.WHITEFLUX
                        time = data.DATE

                        flux_resampled, time_resampled = ssg.resample(flux, sample_size, time)

                        # Creating a new pd.DataFrame
                        concat_dict = {
                        "DATE": pd.Series(time_resampled), 
                        "WHITEFLUX": pd.Series(flux_resampled)
                        }
                        data_resampled = pd.concat(concat_dict, axis=1)

                        # Creating folder with lightcurves resampled
                        if not os.path.isdir(RESAMPLE_PATH):
                            os.mkdir(RESAMPLE_PATH)

                        # Renaming lightcurve
                        file_name = 'RESAMPLED_' + files[j]

                        # Saving lightcurves resampled 
                        FILE_DIR = file_name
                        data_resampled.to_csv(file_name, index=False)
                        # print('Resampled and saved:' + files[j])

                        shutil.move(FILE_DIR, RESAMPLE_PATH)
                        
        # Verify if all files have been resampled                
        path, dirs, files = next(os.walk(RESAMPLE_PATH))
        num_csv_resampled = 0
        for i in files:
            if i.endswith('.csv'):
                num_csv_resampled += 1

        # print("\nTotal of files resampled:", count)
        if num_csv == num_csv_resampled:
            print("All files have been resampled successfully!")
        else:
            print("Not all files have been converted! Uncomment the `print` statement on line 140 to see details")


    @staticmethod
    def compute_periodogram_feature(CSV_PATH: str) -> pd.DataFrame:
        pass

    @staticmethod
    def compute_folded_curve(INPUT_CURVES_PATH: str, OUTPUT_FOLDED_CURVES_PATH: str):
        OUTPUT_FOLDED_CURVES_PATH = OUTPUT_FOLDED_CURVES_PATH.replace("\\", "/")
        total_files = 0
        for root_dir_path, sub_dirs, files in os.walk(INPUT_CURVES_PATH):
            for file in files:
                if file.endswith('.csv'):
                    total_files += 1
        

        with tqdm(range(total_files), colour='blue', desc='Simulating') as pbar:
            for root_dir_path, sub_dirs, files in os.walk(INPUT_CURVES_PATH):
                for file in files:
                    if file.endswith('.csv'):
                        CURVE_PATH = os.path.join(root_dir_path, file)
                        CURVE_PATH = CURVE_PATH.replace("\\", "/")
                        CURVE_ID = CURVE_PATH.split('/')[-1].split('_')[-1].split('.')[0]
                        FILTER_TECHNIQUE = CURVE_PATH.split('/')[6]

                        if FILTER_TECHNIQUE == 'bessel':
                            n = CURVE_PATH.split('/')[-3]
                            f = CURVE_PATH.split('/')[-2]
                            title = f'LC {CURVE_ID}. Bessel {n} and {f}'

                        if FILTER_TECHNIQUE == 'butterworth':
                            n = CURVE_PATH.split('/')[-3]
                            f = CURVE_PATH.split('/')[-2]
                            title = f'LC {CURVE_ID}. Butterworth {n} and {f}'

                        if FILTER_TECHNIQUE == 'gaussian':
                            f = CURVE_PATH.split('/')[-2]
                            title = f'LC {CURVE_ID}. Gaussian {f}'

                        if FILTER_TECHNIQUE == 'ideal':
                            f = CURVE_PATH.split('/')[-2]
                            title = f'LC {CURVE_ID}. Ideal {f}'

                        if FILTER_TECHNIQUE == 'median':
                            numNei = CURVE_PATH.split('/')[-2][-1:]
                            title = f'LC {CURVE_ID}. Median numNei {numNei}'

                        # Reading a curve
                        data = pd.read_csv(CURVE_PATH)
                        curve = LightCurve(data.DATE.to_numpy(), data.WHITEFLUX.to_numpy())
                        # curve.plot(title=title)
                        

                        # Computing folded curve
                        folded_curve = curve.fold(CURVE_ID)
                        # folded_curve.plot(title=f'Folded LC {CURVE_ID}')

                        # Creating a new pd.DataFrame
                        concat_dict = {
                        "TIME": pd.Series(folded_curve.time), 
                        "FOLD_FLUX": pd.Series(folded_curve.flux),
                        "ERROR": pd.Series(folded_curve.flux_error)
                        }

                        folded_data = pd.concat(concat_dict, axis=1)

                        # Saving 
                        SAVING_PATH = OUTPUT_FOLDED_CURVES_PATH + '/' +  '/'.join(CURVE_PATH.split('/')[6:])
                        folded_data.to_csv(SAVING_PATH, index=False)
                        pbar.update(1)

    @staticmethod
    def compute_folded_curve_new(INPUT_CURVES_PATH: str, OUTPUT_FOLDED_CURVES_PATH: str):
        OUTPUT_FOLDED_CURVES_PATH = OUTPUT_FOLDED_CURVES_PATH.replace("\\", "/")
        total_files = 0
        for root_dir_path, sub_dirs, files in os.walk(INPUT_CURVES_PATH):
            for file in files:
                if file.endswith('.csv'):
                    total_files += 1

        with tqdm(range(total_files), colour='blue', desc='Simulating') as pbar:
            for root_dir_path, sub_dirs, files in os.walk(INPUT_CURVES_PATH):
                for file in files:
                    if file.endswith('.csv'):
                        CURVE_PATH = os.path.join(root_dir_path, file)
                        CURVE_PATH = CURVE_PATH.replace("\\", "/")
                        CURVE_ID = CURVE_PATH.split('/')[-1].split('_')[-1].split('.')[0]
                        data = pd.read_csv(CURVE_PATH)
                        curve = LightCurve(data.DATE.to_numpy(), data.WHITEFLUX.to_numpy())
                        folded_curve = curve.fold(CURVE_ID)

                        concat_dict = {
                            "TIME": pd.Series(folded_curve.time),
                            "FOLD_FLUX": pd.Series(folded_curve.flux),
                            "ERROR": pd.Series(folded_curve.flux_error)
                        }

                        folded_data = pd.concat(concat_dict, axis=1)
                        SAVING_PATH = OUTPUT_FOLDED_CURVES_PATH + '/' + CURVE_ID + '.csv'
                        folded_data.to_csv(SAVING_PATH, index=False)
                        pbar.update(1)

        


            