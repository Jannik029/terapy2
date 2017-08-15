import TDSData as tds
import matplotlib.pyplot as plt
from TimeDomainData import TimeDomainData
import numpy as np
from OneLayerSystem import calculate_thickness
from OneLayerSystem import calculate_n


#t, amplitude = tds.TDSData._load_from_txt_file('F:/CloudStation/Uni/terapy2/testdata/2017-03-11T17-58-26.240799-GlassEmitterM1-Reference.txt')
#t_sample, amplitude_sample = tds.TDSData._load_from_txt_file('F:/CloudStation/Uni/terapy2/testdata/2017-03-11T18-02-11.465128-GlassEmitterM1-Cuvette1.txt')

#F:/CloudStation/Uni/terapy2/
#data = tds.TDSData.load_from_txt_file('testdata/2017-03-11T17-58-26.240799-GlassEmitterM1-Reference.txt')
                                      #max_value=2e2, min_value=1e2)

#data = tds.TDSData.load_from_txt_file('testdata/2017-03-11T18-02-11.465128-GlassEmitterM1-Cuvette1.txt')
#data.set_frequency_resolution(0.5)
#data.plot_frequency_domain()
#data.set_frequency_boundaries(300e9, 4e12)
#data.remove_phase_offset()
#data.plot_frequency_domain()
#data.plot_time_domain()
#plt.plot(t, amplitude)
#plt.plot(t_sample, amplitude_sample)


#reference = TimeDomainData.load_from_txt_file('testdata/2017-03-11T17-58-26.240799-GlassEmitterM1-Reference.txt', time_factor=1e-12)
#sample = TimeDomainData.load_from_txt_file('testdata/2017-03-11T18-02-11.465128-GlassEmitterM1-Cuvette1.txt', time_factor=1e-12)

#reference = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_14_342_275.txt', time_factor=1e-12)
#sample = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_15_380_279.txt', time_factor=1e-12)

#reference = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_22_724_290.txt', time_factor=1e-12)
#sample = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_23_718_342.txt', time_factor=1e-12)

reference = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_30_462_337.txt', time_factor=1e-12, min_value=9e-12)
sample = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_31_430_335.txt', time_factor=1e-12, min_value=9e-12)


reference.apply_axis(16e-12, 28e-12, reference.time_step)
reference.plot()

#reference = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_9_705_202.txt', time_factor=1e-12, min_value=9e-12)
#sample = TimeDomainData.load_from_txt_file('C:/Users/lehrj/CloudStation/Uni/T-Age/Messungen/2017-08-04/13-11-44/TD/TD_Data_10_158_261.txt', time_factor=1e-12, min_value=9e-12)

#d = calculate_thickness(sample,
#                        reference,
#                        1100e-6,
#                        300e9, 1.25e12,
#                        300e9, 1.25e12,
#                        7,
#                        thickness_step=20e-6,
#                        thickness_interval=400e-6,
#                        do_plot=True, window_slope=17e-12)
#print(d)

#n = calculate_n(sample,
#                reference,
#                1090e-6,
#                300e9, 1.25e12,
#                300e9, 1.25e12,
#                do_plot=True)

#reference.remove_background(5e-12)
#reference.plot()
#reference.apply_axis(1.3e-10, 2.5e-10, 2.5e-14)
#reference.apply_peak_window(window_type='blackman', plot=True)
#reference.amplitude += 0.5 * np.max(reference.amplitude)
#reference.plot()