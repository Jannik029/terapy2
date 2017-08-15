from TimeDomainData import TimeDomainData
from FrequencyDomainData import FrequencyDomainData
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.optimize import minimize
from scipy import interpolate
import csv
import terapytools as tools
import terapytools

def save_n_to_csv(file, f, n):
    with open(file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        i = 0
        while i < len(f):
            writer.writerow((str(f[i]), str(n[i].real), str(n[i].imag)))
            i += 1


def calculate_thickness(sample_data,
                        reference_data,
                        estimated_thickness,
                        calculation_domain_min,
                        calculation_domain_max,
                        phase_calculation_min,
                        phase_calculation_max,
                        no_pulses,
                        thickness_step=5e-6,
                        thickness_interval=30e-6,
                        window_slope=5e-12,
                        method='tv',
                        frequency_resolution=10e9,
                        time_domain_min=None,
                        time_domain_max=None,
                        time_resolution=None,
                        background_removal_time=5e-12,
                        do_plot=True,
                        plot_output_path=None):

    if do_plot is True:
        fig, ((td_ax, fd_ax), (phase_ax, no_ax), (transfer_function_ax, transfer_function_phase_ax)) = plt.subplots(3, 2)
        fig.set_size_inches(10, 10, forward=True)

    # calculate averages
    sample_td, sample_td_std_dev = TimeDomainData.calculate_average(sample_data)
    reference_td, reference_td_std_dev = TimeDomainData.calculate_average(reference_data)

    # remove background
    sample_td.remove_background(background_removal_time)
    reference_td.remove_background(background_removal_time)

    # apply window
    if do_plot is True: plt.sca(td_ax)

    sample_td.apply_window(window_slope, plot=do_plot)
    reference_td.apply_window(window_slope, plot=do_plot)
#    sample_td.apply_peak_window()
#    reference_td.apply_peak_window()

    if do_plot is True:
        plt.title('Time Domain')
        sample_td.plot(label='sample')
        reference_td.plot(label='reference')
        plt.legend(loc=0)

    # bring to common time axis
    if time_domain_min is not None and time_domain_max is not None and time_resolution is not None:
        sample_data.apply_axis(time_domain_min, time_domain_max, time_resolution)
        reference_data.apply_axis(time_domain_min, time_domain_max, time_resolution)

    # calculate frequency domain data
    sample_fd = sample_td.calculate_frequency_domain(frequency_resolution)
    reference_fd = reference_td.calculate_frequency_domain(frequency_resolution)
    sample_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)
    reference_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)

    # crop frequency domain to calculation domain
    sample_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)
    reference_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)

    if do_plot is True:
        plt.sca(fd_ax)
        plt.title('Frequency Domain')
        sample_fd.plot(label='sample')
        reference_fd.plot(label='reference')
        plt.legend(loc=0)

        plt.sca(phase_ax)
        plt.title('Phase')
        sample_fd.plot_phase(label='sample')
        reference_fd.plot_phase(label='reference')
        plt.legend(loc=0)

    # calculate transfer function
    transfer_function = sample_fd / reference_fd

    if do_plot is True:
        plt.sca(transfer_function_ax)
        plt.title('Transfer Function')
        transfer_function.plot()
        plt.sca(transfer_function_phase_ax)
        plt.title('Transfer Function Phase')
        transfer_function.plot_phase()

    thicknesses = np.arange(estimated_thickness - thickness_interval,
                            estimated_thickness + thickness_interval,
                            thickness_step)

    if method == 'tv':
        if do_plot is True: plt.sca(no_ax)
        total_variations = []
        for d in thicknesses:
            n = _calculate_n(d, transfer_function, no_pulses)
            #plt.plot(transfer_function.axis, n.real)
            #plt.plot(transfer_function.axis, n)
            total_variations.append(np.sum(np.abs(np.diff(n.real)) + np.abs(np.diff(n.imag))))

        total_variations = np.array(total_variations)
        fit_thicknesses = np.arange(thicknesses[0], thicknesses[-2], 0.05e-6)
        fit_function = interpolate.interp1d(thicknesses, total_variations, kind='cubic')
        d_opt = fit_thicknesses[np.argmin(fit_function(fit_thicknesses))]

        if do_plot is True:
            plt.title('Total Variation')
            plt.plot(thicknesses * 1e6, total_variations, 'r+')
            plt.vlines(d_opt * 1e6, np.min(total_variations), np.max(total_variations))
            plt.plot(fit_thicknesses * 1e6, fit_function(fit_thicknesses))

    if method == 'qs':
        if do_plot is True: plt.sca(no_ax)
        ns = []
        for d in thicknesses:
            ns.append(_calculate_n(d, transfer_function, no_pulses))

        #for refractive_index in n:
        #    plt.plot(transfer_function.axis, refractive_index)
        qs = []
        for n in ns:
            QSr = np.fft.fft(n.real - np.mean(n.real))
            QSi = np.fft.fft(n.imag - np.mean(n.real))
            ix = list(range(3, int(len(QSr)/2-1)))
            QSr = QSr[ix]
            QSi = QSi[ix]
            qs.append(np.mean(abs(QSr)) + np.mean(abs(QSi)))
            #plt.plot(QSr)
        plt.plot(thicknesses, qs)

        d_opt = 0

    if do_plot:
        plt.tight_layout()

    if plot_output_path is not None:
        terapytools.ensure_dir(plot_output_path + '/')
        plt.savefig(plot_output_path)

    return d_opt


def calculate_n(sample_data,
                reference_data,
                thickness,
                calculation_domain_min,
                calculation_domain_max,
                phase_calculation_min,
                phase_calculation_max,
                background_removal_time=5e-12,
                window_slope=5e-12,
                window_top=5e-12,
                frequency_resolution=10e9,
                do_plot=True,
                plot_output_path=None):

    if do_plot is True:
        fig = plt.figure()
        fig.set_size_inches(10, 10, forward=True)
        plt.subplot(321)

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

    if do_plot is True:
        sample_td.plot(label='sample')
        reference_td.plot(label='reference')
        plt.legend(loc=0)

    # calculate frequency domain data
    sample_fd = sample_td.calculate_frequency_domain(frequency_resolution)
    reference_fd = reference_td.calculate_frequency_domain(frequency_resolution)
    sample_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)
    reference_fd.remove_phase_offset(phase_calculation_min, phase_calculation_max)

    if do_plot is True:
        plt.subplot(322)
        sample_fd.plot(label='sample')
        reference_fd.plot(label='reference')
        plt.legend(loc=0)
        plt.subplot(323)
        sample_fd.plot_phase(label='sample')
        reference_fd.plot_phase(label='reference')
        plt.legend(loc=0)

    # crop frequency domain to calculation domain
    sample_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)
    reference_fd.apply_axis(calculation_domain_min, calculation_domain_max, frequency_resolution)

    # calculate transfer function
    transfer_function = sample_fd / reference_fd

    n = _calculate_n(thickness,
                     transfer_function,
                     0)

    if do_plot is True:
        plt.subplot(325)
        plt.plot(transfer_function.axis * 1e-12, n.real)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('n')
        plt.subplot(326)
        plt.plot(transfer_function.axis * 1e-12, n.imag)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('kappa')
        plt.tight_layout()

        if plot_output_path is not None:
            terapytools.ensure_dir(plot_output_path + '/')
            plt.savefig(plot_output_path)

    #plt.figure()
    #plt.plot(transfer_function.axis, n.real)

    ########################################
    # save_n_to_csv('n.csv', transfer_function.axis, n)
    tools.save_n_to_txt('n.txt', transfer_function.axis, n)
    return transfer_function.axis, n


def calculate_n_approximate(omega, d, transfer_func):
    n_0 = 1
    n = n_0 - c / (omega * d) * transfer_func.phase
    kappa = c / (omega * d) * \
                (np.log(4 * n * n_0 / (n + n_0)**2) - np.log(np.absolute(transfer_func.amplitude)))
    #plt.figure()
    #plt.plot(transfer_func.axis, n)
    return n, -kappa


def minimize_me(n, omega, transfer_func, d, no_pulses):
    n = n[:int(len(n) / 2)] + 1j * n[int(len(n) / 2):]
    t = theo_transfer_func(omega, n, d, no_pulses)
    return np.sum((transfer_func.amplitude.real - t.real)**2 + (transfer_func.amplitude.imag - t.imag)**2)


def _calculate_n(d, transfer_func, no_pulses):
    omega = 2 * np.pi * transfer_func.axis
    n, kappa = calculate_n_approximate(omega, d, transfer_func)
    n_init = np.hstack((n, kappa))

    n_bound = [1, 20]
    kappa_bound = [-40, 0]
    bounds = np.vstack((np.ones((len(n), 2)) * n_bound,
                        np.ones((len(n), 2)) * kappa_bound))

    fres = minimize(minimize_me, n_init, args=(omega, transfer_func, d, no_pulses), bounds=bounds, options={'gtol': 1e-6, 'disp': False})
    n_total = fres.x
    n_t = n_total[:int(len(n_total) / 2)]
    n_k = n_total[int(len(n_total) / 2):]
    n = n_t + 1j * n_k
    return n


def theo_transfer_func(omega, n, d, no_pulses):
    n0 = 1
    q = ((n0 - n) / (n0 + n) * np.exp(-1j * n * omega * d / c)) ** 2
    FP = (q ** (no_pulses + 1) - 1) / (q - 1)
    FP *= 4 * n0 * n / (n0 + n) ** 2 * np.exp(-1j * (n - n0) * omega * d / c)
    return FP