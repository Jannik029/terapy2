import numpy as np
import csv
from pathlib import Path


def save_n_to_csv(file, f, n):
    """
    Saves the complex refractive index in a csv file.
    :param file: ouput file
    :param f: frequency axis. Array of floats
    :param n: complex refractive index. Array of complex values. Has to be the same size as f.
    :return: None
    """
    if len(f) is not len(n):
        raise Exception('Error while saving n to csv: the size of f (' + str(len(f)) + ') is not equal to the size' +
                        'of n (' + str(len(n)) + ') (' + file + ')')

    ensure_dir(file)

    with open(file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        i = 0
        while i < len(f):
            writer.writerow((str(f[i]), str(n[i].real), str(n[i].imag)))
            i += 1


def save_n_to_txt(file, f, n):
    """
    Saves the complex refractive index in a txt file.
    :param file: output file
    :param f: frequency axis. Array of floats.
    :param n: complex refractive index. Array of complex values. Has to be the same size as f.
    :return: None
    """
    if len(f) is not len(n):
        raise Exception('Error while saving n to txt: the size of f (' + str(len(f)) + ') is not equal to the size' +
                        'of n (' + str(len(n)) + ') (' + file + ')')

    ensure_dir(file)

    np.savetxt(file, np.transpose((f, n.real, n.imag)))


def load_n_from_txt(file):
    """
    Loads the complex refractive index from a txt file.
    :param file: input file containing three columns: f n.real n.imag
    :return: tuple: f array (floats), complex n array (complex)
    """
    data = np.loadtxt(file)
    if data.shape[1] < 3:
        raise Exception('Error while loading n from txt: only ' + str(data.shape[1]) + ' columns found. 3 are required'
                        + '(' + file + ')')

    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def load_csv_table(file, converter, skip_title_row=False):
    """
    Loads a csv file and returns a 2d numpy array.
    :param file: the file which should be loaded
    :param converter: lambda function, which takes one argument and which converts each cell to a user defined format
    :param skip_title_row: if true, the first line in the csv file will be skipped
    :return: a 2d array which has the shape (rows, columns)
    """
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
    """
    Checks, if the given folder is valid. If not, the folder and every parent folders will be created.
    :param file_path: folder or file to be checked.
    :return: None
    """
    path = Path(file_path)
    if '.' in file_path:
        path = path.parent

    path.mkdir(parents=True, exist_ok=True)


def save_to_csv(file, column1, column2):
    """
    Saves to Columns to a csv file.
    :param file: filename of the csv file.
    :param column1: first column
    :param column2: second column
    :return: None
    """
    if len(column1) is not len(column2):
        raise Exception('Error while saving to csv: both columns have to have the same length (' + file + ')')

    ensure_dir(file)
    with open(file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in np.arange(len(column1)):
            writer.writerow([str(column1[i]), str(column2[i])])
