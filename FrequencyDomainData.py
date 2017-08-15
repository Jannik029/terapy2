import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class FrequencyDomainData:

    def __init__(self, frequency_axis, amplitude, unwrapped_phase):

        self.axis = frequency_axis
        self.amplitude = amplitude
        self.phase = unwrapped_phase
        self.frequency_step = self.axis[1] - self.axis[0]
        self.sampling_points = len(self.axis)

    def remove_phase_offset(self, start, end):
        """
        Interpolates the phase to zero and subtracts the offset from the amplitude
        :param start: start of calculation domain
        :param end: end of calculation domain
        """

        ix = np.all(np.vstack((self.axis > start, self.axis < end)), axis=0)
        f = self.axis[ix]
        phase = self.phase[ix]

        offset = np.polyfit(f, phase, 1)
        self.phase -= offset[1]

    def crop(self, frequency_min, frequency_max):
        ix = np.all([self.axis >= frequency_min, self.axis <= frequency_max], axis=0)
        self.axis = self.axis[ix]
        self.amplitude = self.amplitude[ix]
        self.phase = self.phase[ix]

    def apply_axis(self, start, end, step):
        """
        Changes the axis using a linear interpolation between the points.
        :param start: start time of the new frequency domain axis
        :param end: end frequency of the new frequency domain axis (is not included)
        :param step: new time step
        """
        new_axis = np.arange(start, end, step)
        new_amplitude = np.zeros(len(new_axis), dtype=np.complex64)
        new_phase = np.zeros(len(new_axis))

        start_old = self.axis[0]
        end_old = self.axis[-1]
        interpolation_function = interpolate.interp1d(self.axis,
                                                      self.amplitude,
                                                      copy=False,
                                                      assume_sorted=True,
                                                      fill_value=1+0j)
        interpolation_function_phase = interpolate.interp1d(self.axis,
                                                            self.phase,
                                                            copy=False,
                                                            assume_sorted=True,
                                                            fill_value=0)

        i = 0
        while i < len(new_axis):
            t = new_axis[i]
            if start_old <= t <= end_old:
                new_amplitude[i] = interpolation_function(t)
                new_phase[i] = interpolation_function_phase(t)
            i += 1

        self.axis = new_axis
        self.amplitude = new_amplitude
        self.phase = new_phase
        self.frequency_step = step
        self.sampling_points = len(new_axis)

    def get_normalized_spectrum(self):
        return 20*np.log10(abs(self.amplitude)/max(abs(self.amplitude)))

    def plot(self, *plot_arguments, **plot_dictionary):
        plt.plot(self.axis / 1e12, self.get_normalized_spectrum(), *plot_arguments, **plot_dictionary)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Normalized Power (dB)')

    def plot_not_normalized(self, *plot_arguments, **plot_dictionary):
        plt.plot(self.axis / 1e12, self.amplitude, *plot_arguments, **plot_dictionary)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Amplitude')

    def plot_phase(self, *plot_arguments, **plot_dictionary):
        plt.plot(self.axis / 1e12, self.phase, *plot_arguments, **plot_dictionary)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase')

    def __truediv__(self, other):

        if len(other.axis) != len(self.axis) or other.axis[0] != self.axis[0] or other.axis[-1] != self.axis[-1]:
            print('Error dividing two FD')
            return None
        else:
            spectrum = self.amplitude / other.amplitude
            phase = self.phase - other.phase
            return FrequencyDomainData(self.axis, spectrum, phase)

