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
from scipy.signal import find_peaks, peak_widths

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

    def __init__(self, time: np.ndarray = None, flux: np.ndarray = None, flux_error: np.ndarray = None) -> None:
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
            self.flux_error = np.full(
                shape=(len(self.time), ), fill_value=np.nan)

    def __repr__(self) -> str:
        print(tabulate(np.c_[self.time, self.flux, self.flux_error], headers=[
              'Time', 'Flux', 'Flux Error'], tablefmt='fancy_grid'))
        return "LightCurve Object"


class LightCurve(BaseLightCurve):

    def plot(self, title='Lightcurve', x_axis='Julian Data', y_axis='Flux', label='Lightcurve') -> None:
        line_plot(x_data=self.time, y_data=self.flux, title=title,
                  x_axis=x_axis, y_axis=y_axis, label=label)

    def view_fourier_spectrum(self) -> None:
        freq = np.fft.fftfreq(self.flux.shape[-1])
        sp = np.fft.fft(self.flux)

        line_plot(x_data=freq, y_data=np.real(sp), title='Original Fourier Spectrum',
                  x_axis='Frequency', y_axis='Magnitude', label='Spectrum', y_axis_type='log')

    def __apply_filter(self,
                       time: np.ndarray,
                       flux: np.ndarray,
                       filter_technique: str,
                       cutoff_freq: float,
                       order: int,
                       numNei: int,
                       numExpansion=70):  # -> FilteredLightCurve:

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
            array_filtered = remove_expand_edges(
                ifft, numExpansion=numExpansion)
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
                    filter_array[i] = 1.0 / \
                        (1.0+(abs(i-(xc-1.0))/D0)**(2.0*order))

            elif filter_technique.upper() == 'BESSEL':
                # Coef ak
                coef = []
                i = 0
                while i <= order:
                    ak = (factorial(2*order - i)) / \
                        (2**(order - i)*factorial(i)*factorial(order - i))
                    coef.append(ak)
                    i += 1

                # Computing θn(s)
                s = TransferFunction.s
                theta_array = []
                k = 0

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

                i = 0
                for i in range(len_filter):
                    filter_array[i] = np.real(
                        evalfr(G, (np.abs(i-(xc-1.0))/D0)))

            raw_filtered = filter_array * fourier
            ifft_raw_filtered = inverse_fourier_transform(raw_filtered)
            no_padded_ifft_raw_filtered = remove_padding(ifft_raw_filtered)
            no_expanded_no_padded_ifft_raw_filtered = remove_expand_edges(
                no_padded_ifft_raw_filtered, numExpansion)

            array_filtered = centralize_fourier(
                no_expanded_no_padded_ifft_raw_filtered)

            array_filtered += (np.mean(flux) - np.mean(array_filtered))

        return FilteredLightCurve(time=time, flux=flux, flux_error=None, filtered_flux=array_filtered, filter_technique=filter_technique, cutoff_freq=cutoff_freq, order=order, numNei=numNei)

    def ideal_lowpass_filter(self, cutoff_freq, numExpansion: int = 70):
        return self.__apply_filter(self.time, self.flux, filter_technique='ideal', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def gaussian_lowpass_filter(self, cutoff_freq, numExpansion: int = 70):
        return self.__apply_filter(self.time, self.flux, filter_technique='gaussian', cutoff_freq=cutoff_freq, numExpansion=numExpansion, order=None, numNei=None)

    def butterworth_lowpass_filter(self, order, cutoff_freq, numExpansion: int = 70):
        return self.__apply_filter(self.time, self.flux, filter_technique='butterworth', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def bessel_lowpass_filter(self, order, cutoff_freq, numExpansion: int = 100):
        return self.__apply_filter(self.time, self.flux, filter_technique='bessel', order=order, cutoff_freq=cutoff_freq, numExpansion=numExpansion, numNei=None)

    def median_filter(self, numNei, numExpansion: int = 70):
        return self.__apply_filter(self.time, self.flux, filter_technique='median', numNei=numNei, numExpansion=numExpansion, cutoff_freq=None, order=None)

    def fold(self, corot_id: str):
        """codigo velho"""
        # P = LightCurve.get_true_value(corot_id, 'Per')
        # time = np.arange(0, len(self.time))
        # normalized_flux = self.flux / np.median(self.flux)
        # normalized_inverted_curve = LightCurve(
        #     time=time, flux=-1*normalized_flux)
        # x = normalized_inverted_curve.flux

        # # Determine eclipses peaks
        # peaks, _ = find_peaks(x, distance=P*100)
        # peaks = peaks[1:-1]
        # totalEclipses = len(peaks)

        # # Determine eclipses widths
        # widths = peak_widths(x, peaks, rel_height=0.5)
        # eclipse_widths_mean = np.mean(widths[0])

        # Summing all eclipses
        # sum_eclipses_flux = 0
        # for numEclipse in range(totalEclipses):
        #     cond = (normalized_inverted_curve.time > (peaks[numEclipse]-eclipse_widths_mean/2)) & (
        #         normalized_inverted_curve.time < (peaks[numEclipse]+eclipse_widths_mean/2))
        #     sum_eclipses_flux += normalized_inverted_curve.flux[cond]

        # sum_eclipses_flux *= -1

        # # Computing the average eclipse
        # avg_eclipses_flux = sum_eclipses_flux/totalEclipses

        # time_fold = np.arange(-len(avg_eclipses_flux)/2,
        #                       len(avg_eclipses_flux)/2)

        # # Return
        # phase_folded_curve = PhaseFoldedLightCurve(
        #     time=time_fold, flux=avg_eclipses_flux)
        # error = np.std(avg_eclipses_flux)
        # error_array = [error for i in range(len(avg_eclipses_flux))]
        # phase_folded_curve.flux_error = error_array

        # return phase_folded_curve

        """codigo novo"""
        # Ajusts time axis
        time = np.arange(0, len(self.flux))

        # Based on corot_id, load the positions and the widths of eclipses
        positions, width = self.__load_fold_infos(corot_id=corot_id)

        # How many eclipses there are on this curve?
        totalEclipses = len(positions)

        # Summing all eclipses
        sum_eclipses_flux = 0
        for eclipse in range(totalEclipses):
            cond = (time > (positions[eclipse]-width/2)) & (time < (positions[eclipse]+width/2))
            sum_eclipses_flux += self.flux[cond]

        # Computing the average eclipse
        avg_eclipses_flux = sum_eclipses_flux/totalEclipses

        # Return
        time_fold = np.arange(-len(avg_eclipses_flux)/2, +len(avg_eclipses_flux)/2)

        folded_curve = PhaseFoldedLightCurve(time=time_fold, flux=avg_eclipses_flux)
        error = np.std(avg_eclipses_flux) #TODO Revisar esse erro aqui !!
        error_array = [error for i in range(len(avg_eclipses_flux))]
        folded_curve.flux_error = error_array

        return folded_curve
        
    def __load_fold_infos(self, corot_id: str):
        if corot_id == '100725706':
            eclipses_position = [1072, 2471, 3871, 5275, 6679, 8081, 9482, 10888, 12291, 13693]
            eclipses_width = 14

        if corot_id == '101086161':
            eclipses_position = [244, 899, 1554, 2217, 2871, 3532, 4188, 4848, 5505, 6161, 6820, 7480, 8134, 8792, 9455, 10109, 10771, 11428, 12088, 12745, 13402, 14061, 14719]
            eclipses_width = 13

        if corot_id == '101368192':
            eclipses_position = [377, 797, 1220, 1641, 2067, 2486, 2903, 3328, 3747, 4171, 4594, 5012, 5436, 5859, 6274, 6698, 7116, 7542, 7961, 8380, 8805, 9226, 9648, 10068, 10488, 10910, 11334, 11754, 12176, 12602, 13024, 13437, 13862, 14287, 14704]
            eclipses_width = 17

        if corot_id == '102671819':
            eclipses_position = [135, 461, 783, 1106, 1431, 1753, 2078, 2401, 2729, 3049, 3371, 3694, 4019, 4345, 4669, 4989, 5316, 5636, 5963, 6285, 6611, 6933, 7256, 7581, 7906, 8226, 8553, 8875, 9199, 9524, 9846, 10171, 10493, 10819, 11145, 11465, 11790, 12113, 12438, 12763, 13088, 13409, 13735, 14059, 14385, 14757, 15031]
            eclipses_width = 14

        if corot_id == '102764809':
            eclipses_position = [324, 800, 1278, 1754, 2227, 2702, 3180, 3651, 4128, 4604, 5074, 5555, 6028, 6504, 6981, 7455, 7930, 8408, 8882, 9356, 9835, 10355, 10831, 11306, 11782, 12254, 12730, 13207, 13685, 14159, 14632]
            eclipses_width = 16

        if corot_id == '102890318':
            eclipses_position = [69, 485, 898, 1313, 1731, 2144, 2561, 2974, 3392, 3806, 4219, 4637, 5050, 5471, 5882, 6290, 6712, 7120, 7538, 7955, 8368, 8785, 9205, 9618, 10031, 10445, 10862, 11274, 11691, 12108, 12522, 12935, 13350, 13760, 14184, 14598, 15008]
            eclipses_width = 31

        if corot_id == '102912369':
            eclipses_position = [1651, 4051, 6441, 8836, 11246, 13644]
            eclipses_width = 50

        if corot_id == '105209106':
            eclipses_position = [112, 558, 996, 1425, 1863, 2300, 2738, 3168, 3615, 4037, 4481, 4918, 5359, 5795, 6224, 6658, 7094, 7530, 7975, 8408, 8846, 9289, 9714, 10154, 10590, 11029, 11464, 11901, 12770, 13209, 13645, 14086, 14517, 14965]
            eclipses_width = 20

        if corot_id == '105793995':
            eclipses_position = [390, 896, 1402, 1908, 2405, 2906, 3413, 3911, 4415, 4920, 5412, 5917, 6421, 6926, 7433, 7939, 8444, 8952, 9457, 9964, 10466, 10973, 11480, 11985, 12492, 13081, 14010, 14516, 15022]
            eclipses_width = 8
            
        if corot_id == '105819653':
            eclipses_position = [133, 1054, 1845, 2684, 3701, 4709, 5705, 6726, 7736, 8990, 9774, 10785, 11801, 12814, 13836, 14840]
            eclipses_width = 20
        
        if corot_id == '105833549':
            eclipses_position = [131, 443, 753, 1066, 1377, 1686, 1997, 2301, 2615, 2924, 3237, 3546, 3849, 4160, 4469, 4783, 5095, 5389, 5702, 6013, 6324, 6635, 6948, 7260, 7569, 7880, 8194, 8505, 8816, 9127, 9440, 9751, 10065, 10375, 10686, 10999, 11312, 11623, 11935, 12245, 12555, 12868, 13181, 13489, 13802, 14115, 14427, 14737, 15049]
            eclipses_width = 12
        
        if corot_id == '105891283':
            eclipses_position = [3202, 13087]
            eclipses_width = 48
        
        if corot_id == '106017681':
            eclipses_position = [560, 1485, 2400, 3322, 4237, 5144, 6071, 6994, 7920, 8846, 9767, 10695, 11621, 12544, 13466, 14393]
            eclipses_width = 19
        
        if corot_id == '110839339':
            eclipses_position = [27, 457, 993, 1516, 2045, 2581, 3108, 3639, 4166, 4429, 4693, 5224, 5754, 6283, 6814, 7347, 7881, 8404, 8935, 9468, 9999, 10523, 11055, 11586, 12117, 12648, 13175, 13706, 14238, 14764]
            eclipses_width = 17
        
        if corot_id == '315198039':
            eclipses_position = [507, 2933, 5363, 7741, 10174, 12570, 15045]
            eclipses_width = 122

        if corot_id == '315211361':
            eclipses_position = [251, 1416, 2616, 3798, 4967, 6136, 7331, 8498, 9688, 10840, 12028, 13220, 14384]
            eclipses_width = 60

        if corot_id == '315239728':
            eclipses_position = [3404, 9142, 14857]
            eclipses_width = 57

        if corot_id == '652180991':
            eclipses_position = [852, 1778, 2728, 3661, 4595, 5535, 6478, 7417, 8350, 9294, 10228, 11163, 12103, 13039, 13979, 14915]
            eclipses_width = 42
        
        # else:
        #     raise NameError(f'{corot_id} is not a valid CoRoT-ID. Please check it')

        return eclipses_position, eclipses_width

    def __replace_negative_values(array: np.ndarray) -> np.ndarray:
        array[array < 0] = 0
        return array

    def __remove_duplicate_values(array: np.ndarray) -> np.ndarray:
        return np.array(list(set(array)))

    def __replace_zero_values(array: np.ndarray) -> np.ndarray:
        return np.where(array == 0, array[-1], array)

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
            return float(complete_table_5[complete_table_5['CoRoT'] == corot_id][parameter].to_numpy()[0])

    @abstractmethod
    def define_interval_period(period: float) -> np.ndarray:
        period_values = np.arange(
            round(period, 2)-0.02, round(period, 2)+0.03, 0.01)
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
        adivR_values = np.arange(round(adivR, 2)-0.02,
                                 round(adivR, 2)+0.03, 0.01)
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

    @staticmethod
    def export_filters_to_csv(
            WHERE_TO_SAVE_PATH: str,
            WHERE_ARE_THE_RESAMPLED_DATASET: str,
            FILTER_TECHNIQUE: str,
            cutoff_freq_range=None,
            order_range=None,
            numNei_range=None):

        DATASET_PATH = WHERE_ARE_THE_RESAMPLED_DATASET

        total = 18

        if cutoff_freq_range != None:
            cutoff_freqs = np.arange(
                start=cutoff_freq_range[0], stop=cutoff_freq_range[1]+cutoff_freq_range[2], step=cutoff_freq_range[2])
            total *= len(cutoff_freqs)

        if order_range != None:
            orders = np.arange(
                start=order_range[0], stop=order_range[1]+order_range[2], step=order_range[2])
            total *= len(orders)

        if numNei_range != None:
            neighboors = np.arange(
                start=numNei_range[0], stop=numNei_range[1]+numNei_range[2], step=numNei_range[2])
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
                                filtered = curve.ideal_lowpass_filter(
                                    cutoff_freq=cutoff_freq)
                                flux_filtered = filtered.get_filtered_flux()

                                concat_dict = {
                                    'DATE': pd.Series(time),
                                    'WHITEFLUX': pd.Series(flux_filtered)
                                }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + \
                                    '/ideal/f0' + str(int(cutoff_freq*10))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_f0' + \
                                    str(int(cutoff_freq*10)) + '_' + files[j]

                                # Saving data
                                filtered_df.to_csv(file, index=False)
                                pbar.update(1)

                        if FILTER_TECHNIQUE.upper() == 'GAUSSIAN':
                            for cutoff_freq in cutoff_freqs:
                                filtered = curve.gaussian_lowpass_filter(
                                    cutoff_freq=cutoff_freq)
                                flux_filtered = filtered.get_filtered_flux()

                                concat_dict = {
                                    'DATE': pd.Series(time),
                                    'WHITEFLUX': pd.Series(flux_filtered)
                                }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + \
                                    '/gaussian/f0' + str(int(cutoff_freq*10))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_f0' + \
                                    str(int(cutoff_freq*10)) + '_' + files[j]

                                # Saving data
                                filtered_df.to_csv(file, index=False)
                                pbar.update(1)

                        if FILTER_TECHNIQUE.upper() == 'BUTTERWORTH':
                            for cutoff_freq in cutoff_freqs:
                                for order in orders:
                                    filtered = curve.butterworth_lowpass_filter(
                                        order=order, cutoff_freq=cutoff_freq)
                                    flux_filtered = filtered.get_filtered_flux()

                                    concat_dict = {
                                        'DATE': pd.Series(time),
                                        'WHITEFLUX': pd.Series(flux_filtered)
                                    }
                                    filtered_df = pd.concat(
                                        concat_dict, axis=1)

                                    folder = WHERE_TO_SAVE_PATH + '/butterworth/n' + \
                                        str(int(order)) + '/f0' + \
                                        str(int(cutoff_freq*10))
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)

                                    file = folder + '/' + FILTER_TECHNIQUE.lower() + '_n' + str(int(order)) + '_f0' + \
                                        str(int(cutoff_freq*10)) + \
                                        '_' + files[j]

                                    # Saving data
                                    filtered_df.to_csv(file, index=False)
                                    pbar.update(1)

                        if FILTER_TECHNIQUE.upper() == 'BESSEL':
                            for cutoff_freq in cutoff_freqs:
                                for order in orders:
                                    filtered = curve.bessel_lowpass_filter(
                                        order=order, cutoff_freq=cutoff_freq, numExpansion=100)
                                    flux_filtered = filtered.get_filtered_flux()

                                    concat_dict = {
                                        'DATE': pd.Series(time),
                                        'WHITEFLUX': pd.Series(flux_filtered)
                                    }
                                    filtered_df = pd.concat(
                                        concat_dict, axis=1)

                                    folder = WHERE_TO_SAVE_PATH + '/bessel/n' + \
                                        str(int(order)) + '/f0' + \
                                        str(int(cutoff_freq*10))
                                    if not os.path.exists(folder):
                                        os.makedirs(folder)

                                    file = folder + '/' + FILTER_TECHNIQUE.lower() + '_n' + str(int(order)) + '_f0' + \
                                        str(int(cutoff_freq*10)) + \
                                        '_' + files[j]

                                    # Saving data
                                    filtered_df.to_csv(file, index=False)
                                    pbar.update(1)

                        if FILTER_TECHNIQUE.upper() == 'MEDIAN':
                            for numNei in neighboors:
                                filtered_curve = curve.median_filter(
                                    numNei=numNei)
                                flux_filtered = filtered_curve.get_filtered_flux()

                                concat_dict = {
                                    'DATE': pd.Series(time),
                                    'WHITEFLUX': pd.Series(flux_filtered)
                                }
                                filtered_df = pd.concat(concat_dict, axis=1)

                                folder = WHERE_TO_SAVE_PATH + \
                                    '/median/numNei' + str(int(numNei))
                                if not os.path.exists(folder):
                                    os.makedirs(folder)

                                file = folder + '/' + FILTER_TECHNIQUE.lower() + '_num' + \
                                    str(int(numNei)) + '_' + files[j]

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

        multi_line_plot(x_data=self.time, y1_data=self.flux, y2_data=self.filtered_flux,
                        label_y1='Original', label_y2='Filtered', title=title, x_axis='Julian Date', y_axis='Flux')

    def view_fourier_results(self) -> None:
        pass

    def get_filtered_flux(self) -> np.ndarray:
        return self.filtered_flux


class PhaseFoldedLightCurve(LightCurve):
    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_error: np.ndarray = None) -> None:
        super().__init__(time, flux, flux_error)

    def plot(self, title: str = 'Folded LightCurve'):
        super().plot(title=title, label='Folded Lightcurve')


class SimulatedPhaseFoldedLightCurve(BaseLightCurve):
    # Attributes
    simulated_time: np.ndarray
    simulated_flux: np.ndarray
    chi2: float

    def __init__(self, time: np.ndarray = None, flux: np.ndarray = None, flux_error: np.ndarray = None, simulated_time: np.ndarray = None, simulated_flux: np.ndarray = None, chi2: float = None) -> None:
        super().__init__(time=time, flux=flux, flux_error=flux_error)
        self.simulated_time = simulated_time
        self.simulated_flux = simulated_flux
        self.chi2 = chi2

    def __repr__(self) -> str:
        return super().__repr__()

    def view_simulation_results(self):
        print('Plotting simulation results')
        multi_line_plot(x_data=self.time, y1_data=self.flux, y2_data=self.simulated_flux, label_y1='Original',
                        label_y2='Simulated', title='Phase-Folded Comparation', x_axis='Julian Data', y_axis='Flux')

    def compare_results(self, see_values=True) -> float:
        if see_values:
            print(tabulate(np.c_[self.flux, self.simulated_flux], headers=[
                  'Original flux', 'Simulated flux'], tablefmt='fancy_grid'))
        # print('Chi squared =', round(self.chi2, 4))
        return self.chi2
