import matplotlib.pyplot as plt
from TimeDomainData import TimeDomainData
import numpy as np
#from ThreeLayerSystem import calculate_thickness
from ThreeLayerSystem import calculate_n
import csv


def load_n_from_csv(file):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        f = []
        value = []
        for row in reader:
            f.append(float(row[0]))
            value.append(float(row[1]) + 1j * float(row[2]))
        return np.array(f), np.array(value)


reference = TimeDomainData.load_from_txt_file('testdata/three_layer/reference.txt')
sample = TimeDomainData.load_from_txt_file('testdata/three_layer/cuvette.txt')

f_e, n_e = load_n_from_csv('testdata/three_layer/emitter/n.csv')
f_r, n_r = load_n_from_csv('testdata/three_layer/receiver/n.csv')

calculate_n(sample,
            reference,
            (707e-6, 5942e-6, 707e-6),
            n_e, n_r,
            300e9, 1.25e12,
            300e9, 1.25e12)