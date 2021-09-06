from datetime import time
from operator import mul
from imt_lightcurve.help_functions.filter_helper import *
from imt_lightcurve.visualization.data_viz import line_plot, multi_line_plot

# from dataclasses import dataclass, field
from math import exp, factorial
import numpy as np
from control import TransferFunction, evalfr
from scipy.signal import medfilt
from tabulate import tabulate # Fancy compare results


# @dataclass
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
            numExpansion=70):#-> FilteredLightCurve

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


class FilteredLightCurve(BaseLightCurve):
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

        multi_line_plot(x_data=self.time, y1_data=self.flux, y2_data=self.filtered_flux, label_y1='Original', label_y2='Filtered', title=title, x_axis='Julian Data', y_axis='Flux')

    def view_fourier_results(self) -> None:
        pass


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
        print('\nChi squared =', round(self.chi2, 4))
        return self.chi2

    




    
    