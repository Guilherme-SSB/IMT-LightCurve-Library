from imt_lightcurve.help_functions.filter_helper import *
from imt_lightcurve.visualization.data_viz import line_plot, multi_line_plot

import os
from math import exp, factorial
import lightkurve as lk
import numpy as np
import pandas as pd
from control import TransferFunction, evalfr

from scipy.signal import medfilt
from tabulate import tabulate  # Fancy compare results
from tqdm.std import tqdm


class BaseLightCurve():
    # Attributes
    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray=None) -> None:
        if isinstance(time, list):
            self.time = np.array(time, dtype='float64')
            
        if isinstance(flux, list):
            self.flux = np.array(flux, dtype='float64')

        if isinstance(flux_error, list):
            self.flux_error = np.array(flux_error, dtype='float64')

        else:
            self.time = time
            self.flux = flux
            self.flux_error = flux_error


    def __repr__(self) -> str:
        return "LightCurve Object"



class LightCurve(BaseLightCurve):  

    def plot(self, title='Lightcurve', x_axis='Julian Data', y_axis='Flux', label='Lightcurve') -> None:
        line_plot(x_data=self.time, y_data=self.flux, title=title, x_axis=x_axis, y_axis=y_axis, label=label)

    def view_fourier_spectrum(self) -> None:
        freq = np.fft.fftfreq(self.flux.shape[-1])
        sp = np.fft.fft(self.flux)
        
        line_plot(x_data=freq, y_data=np.real(sp), title='Original Fourier Spectrum', x_axis='Frequency', y_axis='Magnitude', label='Spectrum', y_axis_type='log')

    def __apply_filter(self, 
            time: np.ndarray, 
            flux: np.ndarray, 
            filter_technique: str, 
            cutoff_freq: float, 
            order: int, 
            numNei: int, 
            numExpansion=70):# -> FilteredLightCurve:

        if filter_technique.upper() == 'IDEAL':
            # Extracting info from curve
            n_time = len(flux)
            D0 = cutoff_freq*n_time

            # Procedures to apply the ideal lowpass filter
            expanded = expand_edges(flux, numExpansion=numExpansion)
            fourier = fourier_transform(expanded)

            i = 0
            for i in range(len(fourier)):
                if fourier[i] > D0:
                    fourier[i] = 0
            
            ifft = inverse_fourier_transform(fourier)
            array_filtered = remove_expand_edges(ifft, numExpansion=numExpansion)
            array_filtered += (flux.mean() - array_filtered.mean())


        elif filter_technique.upper() == 'MEDIAN':
            array_filtered = medfilt(flux, numNei)
            
        else:
            # Extracting info from curve
            n_time = len(flux)
            D0 = cutoff_freq*n_time
            xc = n_time

            # Procedures to filtering on frequency domain
            expanded = expand_edges(flux, numExpansion=numExpansion)
            padded = padding(expanded)
            centralized = centralize_fourier(padded)
            fourier = fourier_transform(centralized)

            # Creating the low-pass transfer function array
            len_filter = len(fourier)
            filter_array = np.zeros(len_filter)

            if filter_technique.upper() == 'GAUSSIAN':
                i = 0
                for i in range(len_filter):
                    filter_array[i] = exp((-(i-(xc-1.0))**2)/(2*((D0)**2)))

            
            elif filter_technique.upper() == 'BUTTERWORTH':
                i = 0
                for i in range(len_filter):
                    filter_array[i] = 1.0 / (1.0+(abs(i-(xc-1.0))/D0)**(2.0*order))


            elif filter_technique.upper() == 'BESSEL':
                # Coef ak
                coef = []
                i = 0
                while i <= order:
                    ak = (factorial(2*order - i)) / ( 2**(order - i)*factorial(i)*factorial(order - i) )
                    coef.append(ak)
                    i += 1

                # Computing θn(s)
                s = TransferFunction.s
                theta_array = []
                k=0

                for k in range(order+1):
                    theta_n = coef[k] * (s**k)
                    theta_array.append(theta_n)

                # Computing H(s)
                coef_numerator = theta_array[0]
                list_denominator = theta_array[:]
                denominator = 0

                for item in list_denominator:
                    denominator += item

                # Computing Transfer Function
                G = coef_numerator / denominator

                i=0
                for i in range(len_filter):
                    filter_array[i] = np.real(evalfr(G, ( np.abs(i-(xc-1.0))/D0 )))

                
            raw_filtered = filter_array * fourier
            ifft_raw_filtered = inverse_fourier_transform(raw_filtered)
            no_padded_ifft_raw_filtered = remove_padding(ifft_raw_filtered)
            no_expanded_no_padded_ifft_raw_filtered = remove_expand_edges(
                no_padded_ifft_raw_filtered, numExpansion)

            array_filtered = centralize_fourier(no_expanded_no_padded_ifft_raw_filtered)

            array_filtered += (np.mean(flux) - np.mean(array_filtered))

        return FilteredLightCurve(time=time, flux=flux, flux_error=None, filtered_flux=array_filtered, filter_technique=filter_technique, cutoff_freq=cutoff_freq, order=order, numNei=numNei)
        
    def ideal_lowpass_filter(self, cutoff_freq, numExpansion=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='ideal', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def gaussian_lowpass_filter(self, cutoff_freq, numExpansion=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='gaussian', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def butterworth_lowpass_filter(self, order, cutoff_freq, numExpansion=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='butterworth', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def bessel_lowpass_filter(self, order, cutoff_freq, numExpansion=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='bessel', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def median_filter(self, numNei, numExpansion=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='median', numNei=numNei, numExpansion=numExpansion, cutoff_freq=None, order=None)

    def fold(self):
        lightkurve = lk.LightCurve(time=self.time, flux=self.flux)

        # Grid os peridods to search
        period = np.linspace(1, 10, 1000)

        # Create a BLS Periodogram
        bls = lightkurve.to_periodogram('bls', period=period, frequency_factor=500)

        # Extracting info about the BLS Periodogram
        planet_b_period = bls.period_at_max_power
        planet_b_t0 = bls.transit_time_at_max_power
        # print('Period at max power = ', planet_b_period)
        # print('Transit time at max power =', planet_b_t0)

        # Folded parameters
        folded_time = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).time.value

        folded_flux = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).flux.value

        folded_flux_error = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).flux_err.value

        return PhaseFoldedLightCurve(time=folded_time, flux=folded_flux, flux_error=folded_flux_error)

    def export_filters_to_csv(
            self,
            WHERE_TO_SAVE_PATH:str, 
            WHERE_ARE_THE_RESAMPLED_DATASET:str, 
            FILTER_TECHNIQUE:str, 
            cutoff_freq_range=None, 
            order_range=None, 
            numNei_range=None):

        DATASET_PATH = WHERE_ARE_THE_RESAMPLED_DATASET

        total = 33

        if cutoff_freq_range != None:
            cutoff_freqs = np.arange(start=cutoff_freq_range[0], stop=cutoff_freq_range[1]+cutoff_freq_range[2], step=cutoff_freq_range[2])
            total *= len(cutoff_freqs)
        
        if order_range != None:
            orders = np.arange(start=order_range[0], stop=order_range[1]+order_range[2], step=order_range[2])
            total *= len(orders)

        if numNei_range != None:
            neighboors = np.arange(start=numNei_range[0], stop=numNei_range[1]+numNei_range[2], step=numNei_range[2])
            total *= len(neighboors)



        with tqdm(range(total), colour='blue', desc='Saving') as pbar:
            for root_dir_path, sub_dirs, files in os.walk(DATASET_PATH):
                for j in range(0, len(files)):
                    if files[j].endswith('.csv'):
                        # print(files[j] + ' => Save it!')
                        data = pd.read_csv(root_dir_path+'/'+files[j])
                        time = data.DATE.to_numpy()
                        flux = data.WHITEFLUX.to_numpy()
                        curve = LightCurve(time, flux)

                        if FILTER_TECHNIQUE.upper() == 'IDEAL':
                            for cutoff_freq in cutoff_freqs:
                                filtered = curve.ideal_lowpass_filter(cutoff_freq=cutoff_freq)
                                flux_filtered = filtered.get_filtered_flux()

                                concat_dict = {
                                    'DATE': pd.Series(time),
                                    'WHITEFLUX': pd.Series(flux_filtered)
                                }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + '/ideal/f0' + str(int(cutoff_freq*10))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_f0' + str(int(cutoff_freq*10)) + '_' + files[j]

                                # Saving data
                                filtered_df.to_csv(file, index=False)
                                pbar.update(1)
                        
                        if FILTER_TECHNIQUE.upper() == 'GAUSSIAN':
                            for cutoff_freq in cutoff_freqs:
                                filtered = curve.gaussian_lowpass_filter(cutoff_freq=cutoff_freq)
                                flux_filtered = filtered.get_filtered_flux()

                                concat_dict = {
                                    'DATE': pd.Series(time),
                                    'WHITEFLUX': pd.Series(flux_filtered)
                                }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + '/gaussian/f0' + str(int(cutoff_freq*10))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_f0' + str(int(cutoff_freq*10)) + '_' + files[j]

                                # Saving data
                                filtered_df.to_csv(file, index=False)
                                pbar.update(1)
                    
                        if FILTER_TECHNIQUE.upper() == 'BUTTERWORTH':
                            for cutoff_freq in cutoff_freqs:
                                for order in orders:
                                    filtered = curve.butterworth_lowpass_filter(order=order, cutoff_freq=cutoff_freq)
                                    flux_filtered = filtered.get_filtered_flux()

                                    concat_dict = {
                                        'DATE': pd.Series(time),
                                        'WHITEFLUX': pd.Series(flux_filtered)
                                    }
                                    filtered_df = pd.concat(concat_dict, axis=1)

                                    folder = WHERE_TO_SAVE_PATH + '/butterworth/n' + str(int(order)) + '/f0' + str(int(cutoff_freq*10))
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)

                                    file = folder + '/' + FILTER_TECHNIQUE.lower() + '_n' + str(int(order)) + '_f0' + str(int(cutoff_freq*10)) + '_' + files[j]

                                    # Saving data
                                    filtered_df.to_csv(file, index=False)
                                    pbar.update(1)
                            
                        if FILTER_TECHNIQUE.upper() == 'BESSEL':
                            for cutoff_freq in cutoff_freqs:
                                for order in orders:
                                    filtered = curve.bessel_lowpass_filter(order=order, cutoff_freq=cutoff_freq, numExpansion=100)
                                    flux_filtered = filtered.get_filtered_flux()

                                    concat_dict = {
                                        'DATE': pd.Series(time),
                                        'WHITEFLUX': pd.Series(flux_filtered)
                                    }
                                    filtered_df = pd.concat(concat_dict, axis=1)

                                    folder = WHERE_TO_SAVE_PATH + '/bessel/n' + str(int(order)) + '/f0' + str(int(cutoff_freq*10))
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)

                                    file = folder + '/' + FILTER_TECHNIQUE.lower() + '_n' + str(int(order)) + '_f0' + str(int(cutoff_freq*10)) + '_' + files[j]

                                    # Saving data
                                    filtered_df.to_csv(file, index=False)
                                    pbar.update(1)
                            
                        if FILTER_TECHNIQUE.upper() == 'MEDIAN':
                            for numNei in neighboors:
                                filtered_curve = curve.median_filter(numNei=numNei)
                                flux_filtered = filtered_curve.get_filtered_flux()

                                concat_dict = {
                                        'DATE': pd.Series(time),
                                        'WHITEFLUX': pd.Series(flux_filtered)
                                    }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + '/median/numNei' + str(int(numNei))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)
                    
                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_num' + str(int(numNei)) + '_' + files[j]

                                # Saving data
                                filtered_df.to_csv(file, index=False)
                                pbar.update(1)

        print(f'\nData from {FILTER_TECHNIQUE} has been saved successfully!')
    


class FilteredLightCurve(LightCurve):
    # Attributes
    filtered_flux: np.ndarray
    filter_technique: str
    cutoff_freq: float
    order: int
    numNei: int

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, filtered_flux: np.ndarray, filter_technique: str, cutoff_freq: float, order: int, numNei: int) -> None:
        super().__init__(time, flux, flux_error=flux_error)
        self.filtered_flux = filtered_flux
        self.filter_technique = filter_technique
        self.cutoff_freq = cutoff_freq
        self.order = order 
        self.numNei = numNei
    
    def view_filtering_results(self) -> None:

        if self.filter_technique.upper() == 'IDEAL':
            title = f"{self.filter_technique.capitalize()} filter with Cutoff frequency = {self.cutoff_freq}"

        elif self.filter_technique.upper() == 'MEDIAN':
            title = f"{self.filter_technique.capitalize()} filter with {self.numNei} neighbors"

        elif self.filter_technique.upper() == 'GAUSSIAN':
            title = f"{self.filter_technique.capitalize()} filter with Cutoff frequency = {self.cutoff_freq}"

        elif self.filter_technique.upper() == 'BUTTERWORTH':
            title = f"{self.filter_technique.capitalize()} filter with Order = {self.order} and Cutoff frequency = {self.cutoff_freq}"

        elif self.filter_technique.upper() == 'BESSEL':
            title = f"{self.filter_technique.capitalize()} filter with Order = {self.order} and Cutoff frequency = {self.cutoff_freq}"

        multi_line_plot(x_data=self.time, y1_data=self.flux, y2_data=self.filtered_flux, label_y1='Original', label_y2='Filtered', title=title, x_axis='Julian Date', y_axis='Flux')

    def view_fourier_results(self) -> None:
        pass

    def get_filtered_flux(self) -> np.ndarray:
        return self.filtered_flux
    


class PhaseFoldedLightCurve(LightCurve):
    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray) -> None:
        super().__init__(time, flux, flux_error=flux_error)

    def plot(self):
        super().plot(title='Folded LightCurve', label='Folded Lightcurve')



class SimulatedPhaseFoldedLightCurve(BaseLightCurve):
    # Attributes
    simulated_time: np.ndarray
    simulated_flux: np.ndarray
    chi2: float

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray, simulated_time: np.ndarray, simulated_flux: np.ndarray, chi2: float) -> None:
        super().__init__(time, flux, flux_error=flux_error)
        self.simulated_time = simulated_time
        self.simulated_flux = simulated_flux
        self.chi2 = chi2

    def __repr__(self) -> str:
        return super().__repr__()

    def view_simulation_results(self):
        print('Plotting simulation results')
        multi_line_plot(x_data=self.time, y1_data=self.flux, y2_data=self.simulated_flux, label_y1='Original', label_y2='Simulated', title='Phase-Folded Comparation', x_axis='Julian Data', y_axis='Flux')

    def compare_results(self, see_values=True) -> float:
        if see_values:
            print(tabulate(np.c_[self.flux, self.simulated_flux], headers=['Original flux', 'Simulated flux'], tablefmt='fancy_grid'))
        # print('Chi squared =', round(self.chi2, 4))
        return self.chi2

    




    
    