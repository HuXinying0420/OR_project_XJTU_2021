import numpy as np
import linecache
# import os
cols = 48 # number of column
divided_ch = ',' # divided_character between numbers

def dat_to_matrix(filename):
    cols = 48 # number of column
    divided_ch = ',' # divided_character between numbers
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)
    datamat = np.zeros((rows, cols))
    row = 0

    for line in lines:
        line = line.strip().split(divided_ch) # strip remove block space in line
        datamat[row, :] = line[:]
        row += 1

    return datamat


