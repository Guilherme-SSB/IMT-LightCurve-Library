from imt_lightcurve.help_functions.filter_helper import *
from imt_lightcurve.visualization.data_viz import line_plot, multi_line_plot

from abc import abstractmethod
from shutil import Error
import os
from math import exp, factorial
import lightkurve as lk
import numpy as np
import pandas as pd
from control import TransferFunction, evalfr

from scipy.signal import medfilt
from tabulate import tabulate  # Fancy compare results
from tqdm.std import tqdm

COMPLETE_TABLE_V_DELEUIL_PATH = 'files/asu.tsv'
complete_table_5 = pd.read_csv(COMPLETE_TABLE_V_DELEUIL_PATH, delimiter=';')

class BaseLightCurve():
    # Attributes
    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray

    def __init__(self, time: np.ndarray=None, flux: np.ndarray=None, flux_error: np.ndarray=None) -> None:
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

        if self.flux_error is None:
            self.flux_error = np.full(shape=(len(self.time), ), fill_value=np.nan)
                

    def __repr__(self) -> str:
        print(tabulate(np.c_[self.time, self.flux, self.flux_error], headers=['Time', 'Flux', 'Flux Error'], tablefmt='fancy_grid'))
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

    def ideal_lowpass_filter(self, cutoff_freq, numExpansion: int=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='ideal', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def gaussian_lowpass_filter(self, cutoff_freq, numExpansion: int=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='gaussian', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def butterworth_lowpass_filter(self, order, cutoff_freq, numExpansion: int=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='butterworth', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def bessel_lowpass_filter(self, order, cutoff_freq, numExpansion: int=100):
        return self.__apply_filter(self.time, self.flux, filter_technique='bessel', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def median_filter(self, numNei, numExpansion: int=70):
        return self.__apply_filter(self.time, self.flux, filter_technique='median', numNei=numNei, numExpansion=numExpansion, cutoff_freq=None, order=None)

    def fold(self, smooth_curve: bool=False, window: float=0.15, window_filter: int=201, order_filter: int=3):
        lightkurve = lk.LightCurve(time=self.time, flux=self.flux)

        # Grid os peridods to search
        period = np.linspace(1, 10, 1000)

        # Create a BLS Periodogram
        bls = lightkurve.to_periodogram('bls', period=period, frequency_factor=500)

        # Extracting info about the BLS Periodogram
        planet_b_period = bls.period_at_max_power
        planet_b_t0 = bls.transit_time_at_max_power

        # Folded parameters
        folded_time = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).time.value
        folded_flux = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).flux.value
        # folded_flux_error = lightkurve.flatten().normalize().fold(period=planet_b_period, epoch_time=planet_b_t0).flux_err.value

        # Windowing curve
        time_w = folded_time[(folded_time > -1*window) & (folded_time < window)]
        flux_w = folded_flux[(folded_time > -1*window) & (folded_time < window)]

        if smooth_curve:
            # Smoothing curve
            smoothed_flux = self.__savitzky_golay(flux_w, window_size=window_filter, order=order_filter)

            # Uncertainties
            folded_flux_error = np.std(smoothed_flux)
            folded_flux_error_array = [folded_flux_error for i in range(len(time_w))]

            # Return
            returnCurve = PhaseFoldedLightCurve(time=time_w, flux=smoothed_flux)
            returnCurve.flux_error = folded_flux_error_array
            return returnCurve

        # Uncertainties
        folded_flux_error = np.std(flux_w)
        folded_flux_error_array = [folded_flux_error for i in range(len(time_w))]

        # Return
        returnCurve = PhaseFoldedLightCurve(time=time_w, flux=flux_w)
        # returnCurve = PhaseFoldedLightCurve(time=folded_time, flux=folded_flux)
        returnCurve.flux_error = folded_flux_error_array
        return returnCurve

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

    def __savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except ValueError as err:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    def __replace_negative_values(array: np.ndarray) -> np.ndarray:
        array[array < 0] = 0
        return array

    def __remove_duplicate_values(array: np.ndarray) -> np.ndarray:
        return np.array(list(set(array)))

    def __replace_zero_values(array: np.ndarray) -> np.ndarray:
        return np.where(array==0, array[-1], array)

    @abstractmethod
    def get_true_value(corot_id: int, parameter: str) -> float:
        """The keywords to be passed as `parameter: str` are:
        - 'Per'   -> returns the orbital period
        - 'Rp/R*' -> returns the radius value of the planet compared to the star
        - 'a/R*'  -> returns the orbital radius values compared to star radius
        - 'b'     -> returns the transit impact parameter
        
        """
        if any(complete_table_5['CoRoT'] == corot_id) == False:
            raise Error(f'Not found CoRoT-ID: {corot_id} at Table 5')
        else:
            # return float(complete_table_5[complete_table_5['CoRoT'] == corot_id][parameter].values[0].split('±')[0])
            return float(complete_table_5[complete_table_5['CoRoT'] == corot_id][parameter].to_numpy()[0])

    @abstractmethod
    def define_interval_period(period: float) -> np.ndarray:
        period_values = np.arange(round(period, 2)-0.02, round(period, 2)+0.03, 0.01)
        period_values = LightCurve.__replace_negative_values(period_values)
        period_values = LightCurve.__remove_duplicate_values(period_values)
        period_values = LightCurve.__replace_zero_values(period_values)
        period_values = np.around(period_values, 4)
        period_values = sorted(period_values)
        return np.array(period_values)

    @abstractmethod
    def define_interval_p(p: float) -> np.ndarray:
        p_values = np.arange(round(p, 2)-0.02, round(p, 2)+0.03, 0.01)
        p_values = LightCurve.__replace_negative_values(p_values)
        p_values = LightCurve.__remove_duplicate_values(p_values)
        p_values = LightCurve.__replace_zero_values(p_values)
        p_values = np.around(p_values, 4)
        p_values = sorted(p_values)
        return np.array(p_values)

    @abstractmethod 
    def define_interval_adivR(adivR: float) -> np.ndarray:
        adivR_values = np.arange(round(adivR, 2)-0.02, round(adivR, 2)+0.03, 0.01)
        adivR_values = LightCurve.__replace_negative_values(adivR_values)
        adivR_values = LightCurve.__remove_duplicate_values(adivR_values)
        adivR_values = LightCurve.__replace_zero_values(adivR_values)
        adivR_values = np.around(adivR_values, 4)
        adivR_values = sorted(adivR_values)
        return np.array(adivR_values)

    @abstractmethod
    def define_interval_b(b: float) -> np.ndarray:
        b_values = np.arange(round(b, 2)-0.02, round(b, 2)+0.03, 0.01)
        b_values = LightCurve.__replace_negative_values(b_values)
        b_values = LightCurve.__remove_duplicate_values(b_values)
        b_values = LightCurve.__replace_zero_values(b_values)
        b_values = np.around(b_values, 4)
        b_values = sorted(b_values)
        return np.array(b_values)

   

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
    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray=None) -> None:
        super().__init__(time, flux, flux_error)

    def plot(self):
        super().plot(title='Folded LightCurve', label='Folded Lightcurve')



class SimulatedPhaseFoldedLightCurve(BaseLightCurve):
    # Attributes
    simulated_time: np.ndarray
    simulated_flux: np.ndarray
    chi2: float

    def __init__(self, time: np.ndarray=None, flux: np.ndarray=None, flux_error: np.ndarray=None, simulated_time: np.ndarray=None, simulated_flux: np.ndarray=None, chi2: float=None) -> None:
        super().__init__(time=time, flux=flux, flux_error=flux_error)
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

    

