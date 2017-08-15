from TimeDomainData import TimeDomainData
from FrequencyDomainData import FrequencyDomainData
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.optimize import minimize
from scipy import interpolate

n_0 = 1

def calculate_n(sample_data,
                reference_data,
                thicknesses,
                n_1, n_3,
                calculation_domain_min,
                calculation_domain_max,
                phase_calculation_min,
                phase_calculation_max,
                background_removal_time=5e-12,
                window_slope=5e-12,
                window_top=5e-12,
                frequency_resolution=10e9,
                do_plot=True):

    # calculate averages
    sample_td, sample_td_std_dev = TimeDomainData.calculate_average(sample_data)
    reference_td, reference_td_std_dev = TimeDomainData.calculate_average(reference_data)

    # remove background
    sample_td.remove_background(background_removal_time)
    reference_td.remove_background(background_removal_time)

    # apply window
    sample_td.apply_peak_window(window_length_time=window_slope * 2 + window_top,
                                window_slope_time=window_slope,
                                plot=do_plot)
    reference_td.apply_peak_window(window_length_time=window_slope * 2 + window_top,
                                   window_slope_time=window_slope,
                                   plot=do_plot)

#    sample_td.plot(label='sample')
#    reference_td.plot(label='reference')
#    plt.legend(loc=0)

    # calculate frequency domain data
    sample_fd = sample_td.calculate_frequency_domain(frequency_resolution)
    reference_fd = reference_td.calculate_frequency_domain(frequency_resolution)
    sample_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)
    reference_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)

#    plt.figure()
#    sample_fd.plot(label='sample')
#    reference_fd.plot(label='reference')
#    plt.legend(loc=0)

#    plt.figure()
#    sample_fd.plot_phase(label='sample')
#    reference_fd.plot_phase(label='reference')
#    plt.legend(loc=0)

    # crop frequency domain to calculation domain
    sample_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)
    reference_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)

    # calculate transfer function
    transfer_function = sample_fd / reference_fd

    n = _calculate_n(transfer_function, sample_td, reference_td, n_1, n_3, thicknesses[0], thicknesses[1], thicknesses[2])
    plt.figure(1)
    plt.plot(transfer_function.axis, n.real)
#    plt.title('calculated n.real')
    plt.figure(2)
    plt.plot(transfer_function.axis, n.imag)
#    plt.title('calculated n.imag')

    return transfer_function.axis, n


def p(omega, n, d):
    return np.exp((-1j * omega * n * d) / c)


def theo_transfer_func(omega, n_1, n_2, n_3, d_1, d_2, d_3):
    res = 16 * (n_0 * n_1 * n_2 * n_3) / ((n_0 + n_1) * (n_1 + n_2) * (n_2 + n_3) * (n_3 + n_0))
    res *= p(omega, n_1, d_1)
    res *= p(omega, n_2, d_2)
    res *= p(omega, n_3, d_3)
    res *= p(omega, -n_0, d_1 + d_2 + d_3)
    return res


def minimize_me(n_2, omega, transfer_func, n_1, n_3, d_1, d_2, d_3):
    n_2 = n_2[:int(len(n_2) / 2)] + 1j * n_2[int(len(n_2) / 2):]
    n_1 = n_1[:int(len(n_1) / 2)] + 1j * n_1[int(len(n_1) / 2):]
    n_3 = n_3[:int(len(n_3) / 2)] + 1j * n_3[int(len(n_3) / 2):]
    t = theo_transfer_func(omega, n_1, n_2, n_3, d_1, d_2, d_3)
    return np.sum((transfer_func.amplitude.real - t.real)**2 + (transfer_func.amplitude.imag - t.imag)**2)


def calculate_n_approximate(omega, sample_td, reference_td, n_1, n_3, d_1, d_2, d_3):
    delta_t = sample_td.get_peak_position() - reference_td.get_peak_position()
    n = 1 / d_2 * (c * delta_t - d_1 * (n_1 - n_0) - d_3 * (n_3 - n_0)) + n_0
    peak_coefficient = sample_td.get_peak_to_peak_amplitude() / reference_td.get_peak_to_peak_amplitude()
    kappa = -1 / d_2 * c / omega * np.log(peak_coefficient) - n_1.imag * d_1 / d_2 - n_3.imag * d_3 / d_2

#    plt.figure()
#    plt.title('n.real approx')
#    plt.plot(omega, n)
#    plt.figure()
#    plt.figure('n.imag approx')
#    plt.plot(omega, kappa)

    return n, -kappa


def _calculate_n(transfer_func, sample_td, reference_td, n_1, n_3, d_1, d_2, d_3):
    omega = 2 * np.pi * transfer_func.axis
    n, kappa = calculate_n_approximate(omega, sample_td, reference_td, n_1, n_3, d_1, d_2, d_3)
    n_init = np.hstack((n, kappa))
    n_1 = np.hstack((n_1.real, n_1.imag))
    n_3 = np.hstack((n_3.real, n_3.imag))
    n_bound = [1, 20]
    kappa_bound = [-40, 0]
    bounds = np.vstack((np.ones((len(n), 2)) * n_bound,
                        np.ones((len(n), 2)) * kappa_bound))

    fres = minimize(minimize_me,
                    n_init,
                    bounds=bounds,
                    args=(omega, transfer_func, n_1, n_3, d_1, d_2, d_3),
                    options={'disp': False})

    n_total = fres.x
    n = n_total[:int(len(n_total) / 2)]
    kappa = n_total[int(len(n_total) / 2):]
    return n + 1j * kappa
