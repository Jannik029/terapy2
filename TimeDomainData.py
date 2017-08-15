import numpy as np
import matplotlib.pyplot as plt
from FrequencyDomainData import FrequencyDomainData
from scipy import interpolate
from scipy import signal

class TimeDomainData:

    @staticmethod
    def load_from_txt_file(file_path, time_factor = 1e-12, min_value=None, max_value=None, usecols=None, delimiter=None):
        """
        loads the object from a file
        :param file_path: path to the file or list of files, which contain the time domain data
        :param time_factor: factor to multiply to time axis, to get seconds
        :param min_value: minimum time value
        :param max_value: maximum time value
        :return a list of TimeDomainData objects if a list of files was given or a single TimeDomainData object
        """

        if isinstance(file_path, (list, tuple, np.ndarray)):
            file_list = file_path
        else:
            file_list = [file_path]

        result = []
        for file in file_list:
            if delimiter is not None:
                raw = np.loadtxt(file, usecols=usecols, delimiter=delimiter)
            else:
                raw = np.loadtxt(file, usecols=usecols)
            if min_value is not None:
                raw = raw[raw[:, 0] >= min_value / time_factor]
            if max_value is not None:
                raw = raw[raw[:, 0] < max_value / time_factor]
            result.append(TimeDomainData(raw[:, 0] * time_factor,
                                         raw[:, 1]))

        if len(result) == 1:
            return result[0]
        else:
            return result

    @staticmethod
    def load_from_huebner_file(file_path, time_factor=1e-12):

        if isinstance(file_path, (list, tuple, np.ndarray)):
            file_list = file_path
        else:
            file_list = [file_path]

        result = []
        for file_name in file_list:
            with open(file_name) as file:
                content = file.readlines()

            content_values = []
            for line in content:
                raw_values = line.split('	')

                values = np.zeros(len(raw_values))
                i = 0
                while i < len(values):
                    values[i] = float(raw_values[i].strip('\n'))
                    i += 1
                content_values.append(values)
            raw = np.asarray(content_values)
            result.append(TimeDomainData(raw[:, 0], raw[:, 1]))

        if len(result) == 1:
            return result[0]
        else:
            return result


    @staticmethod
    def calculate_average(time_domains):
        if isinstance(time_domains, (list, tuple, np.ndarray)):
            td_list = time_domains
        else:
            td_list = [time_domains]

        axis = td_list[0].axis
        amplitudes = [td.amplitude for td in td_list]
        amplitude = np.average(amplitudes, axis=0)
        std_deviation = np.std(amplitudes, axis=0, ddof=1) / np.sqrt(len(td_list))
        return TimeDomainData(axis, amplitude), std_deviation

    def __init__(self, time_axis, e_field):

        self.axis = time_axis
        self.amplitude = e_field
        self.time_step = self.axis[1] - self.axis[0]
        self.sampling_points = len(self.axis)

    def remove_background(self, time):
        """
        Calculates the average value of the given time domain and subtracts it from the amplitude
        :param time: to which time the average should be calculated
        """
        steps = int(time / self.time_step)
        average = np.average(self.amplitude[0:steps])
        self.amplitude -= average

    def apply_window(self, window_length_time=5e-12, window_type='tukey', plot=False):
        """
        Applies a window of the given type to the time domain.
        :param window_length_time: length of the window
        :param window_type: currently supported is the tukey window. The window_length_time parameter is the
                            length of one side.
        :param plot: plots the window scaled to the maximum value of amplitude
        """

        if window_type == 'tukey':
            window = signal.tukey(self.sampling_points,
                                  (2 * window_length_time / self.time_step) / self.sampling_points)
            self.amplitude *= window
            if plot is True:
                plt.plot(self.axis * 1e12, window * np.max(self.amplitude))
        else:
            pass

    def apply_peak_window(self, window_length_time=15e-12, window_slope_time=5e-12, window_type='tukey', plot=False):
        """
        Applies a window of the given type to the time domain around the peak.
        :param window_length_time: length of the window
        :param window_type: window type. Can be tukey and blackman.
        :param plot: plots the window scaled to the maximum value of amplitude
        """

        peak_position = np.argmax(self.amplitude)

        if window_type == 'tukey':
            window_N = int(window_length_time / self.time_step)
            window = signal.tukey(window_N,
                                  (2 * window_slope_time / self.time_step) / window_N)
            window = np.concatenate((np.zeros(peak_position - window_N // 2),
                                     window,
                                     np.zeros(self.sampling_points - window_N - peak_position + window_N // 2)))

        elif window_type == 'blackman':
            window_N = int(window_slope_time // self.time_step)
            window = signal.blackman(window_N)
            window = np.concatenate((np.zeros(peak_position - window_N // 2),
                                     window,
                                     np.zeros(self.sampling_points - window_N - peak_position + window_N // 2)))

        else:
            window = np.ones(self.sampling_points)

        self.amplitude *= window
        if plot is True:
            plt.plot(self.axis * 1e12, window * np.max(self.amplitude))

    def get_peak_position(self):
        return self.axis[np.argmax(self.amplitude)]

    def get_peak_to_peak_amplitude(self):
        return np.abs(np.max(self.amplitude)) + np.abs(np.min(self.amplitude))

    def apply_axis(self, start, end, step):
        """
        Changes the axis using a linear interpolation between the points.
        :param start: start time of the new time domain axis
        :param end: end time of the new time domain axis (is not included)
        :param step: new time step
        """
        new_axis = np.arange(start, end, step)
        new_amplitude = np.zeros(len(new_axis))

        start_old = self.axis[0]
        end_old = self.axis[-1]
        interpolation_function = interpolate.interp1d(self.axis,
                                                      self.amplitude,
                                                      copy=False,
                                                      assume_sorted=True,
                                                      fill_value=0,
                                                      bounds_error=False)
        i = 0
        while i < len(new_axis):
            t = new_axis[i]
            if start_old <= t <= end_old:
                new_amplitude[i] = interpolation_function(t)
            i += 1

        self.axis = new_axis
        self.amplitude = new_amplitude
        self.time_step = step
        self.sampling_points = len(new_axis)

    def calculate_frequency_domain(self, resolution=None):
        """
        Creates an FrequencyDomainData object
        :param resolution: Distance between two points in the FD on the frequency axis. If None the
                           frequency resolution will be 1/(time_step * sampling_points
        :return: FrequencyDomainData object
        """
        if resolution is None:
            sampling_points = self.sampling_points
        else:
            sampling_points = int(1 / (self.time_step * resolution))

        f = np.fft.rfftfreq(sampling_points, self.time_step)
        amplitude = np.fft.rfft(self.amplitude, sampling_points)
        phase = np.unwrap(np.angle(amplitude))
        return FrequencyDomainData(f, amplitude, phase)

    def plot(self, *plot_arguments, **plot_dictionary):
        plt.plot(self.axis * 1e12, self.amplitude, *plot_arguments, **plot_dictionary)
        plt.xlabel('Time (ps)')
        plt.ylabel('Amplitude (arb. units)')

