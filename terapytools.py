import numpy as np
import csv
from pathlib import Path


def save_n_to_txt(file, f, n):
    """
    Saves the complex refractive index in a txt file.
    :param file: output file
    :param f: frequency axis. Array of floats.
    :param n: complex refractive index. Array of complex values. Has to be the same size as f.
    """
    ensure_dir(file)
    np.savetxt(file, np.transpose((f, n.real, n.imag)))


def load_n_from_txt(file):
    """
    Loads the complex refractive index from a txt file.
    :param file: input file containing three columns: f n.real n.imag
    :return: tuple: f array (floats), complex n array (complex)
    """
    data = np.loadtxt(file)
    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def load_csv_table(file, converter, skip_title_row=False):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result = []
        i = 0
        for row in reader:
            if skip_title_row is True and i > 0:
                row_result = []
                j = 0
                for item in row:
                    row_result.append(converter[j](item))
                    j += 1
                result.append(row_result)
            elif skip_title_row is False:
                row_result = []
                j = 0
                for item in row:
                    row_result.append(converter[j](item))
                    j += 1
                result.append(row_result)
            i += 1
        return np.array(result)


def ensure_dir(file_path):
    path = Path(file_path)
    if '.' in file_path:
        path = path.parent

    path.mkdir(parents=True, exist_ok=True)


def save_to_csv(file, column1, column2):
    ensure_dir(file)
    with open(file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in np.arange(len(column1)):
            writer.writerow([str(column1[i]), str(column2[i])])
