################################################
# Insert lines to fullfill the standard sequence
#       considering the CSI timestamps         #
################################################
import matplotlib.pyplot as plt
import itertools
import csv
import os
import pandas
import numpy as np
import math
import cmath

lent = []
tmin = []
tsamp = []


NS = 5
N10s = 125

#invalid_rows = [0,4,5,5,4,5,5, 5, 5, 5, 5, 5, 4, 5, 3, 5,0,3,3,5,5] #empty for 1-20 in 326
#invalid_rows = [0,5,4,5,5,5,5, 5, 4, 5, 5, 5, 5, 4, 5, 4,4,4,5,4,4] #walk for 1-20 in 326
#invalid_rows = [0,5,4,5,5,5,4, 4, 4, 5, 3, 5, 5, 5, 5, 4,4,4,4,5,4,4] #sit for 1-21 in 326
#invalid_rows = [0,5,5,4,5,5,5, 4, 4, 5, 5, 5, 4, 0, 5, 5,5,5,5,3,4,5] #stand for 1-21 in 326
#invalid_rows = [0,3,4,5,5,3,4, 5, 5, 4, 5, 0, 4, 4, 4, 4,5,5,5,5,5,4,5] #empty for 1-22 in 127
#invalid_rows = [0,4,4,4,5,3,5, 3, 1, 5, 4, 5, 3, 3, 5, 0,4,5,5,5,5,4,4,5] #sit for 1-23 in 127
#invalid_rows = [0,5,4,4,4,5,5, 4, 0, 4, 4, 4, 5, 3, 4, 4,5,5,4,5,5,5,5,5] #stand for 1-23 in 127
#invalid_rows = [0,5,0,4,5,4,5, 5, 5, 3, 4, 5, 4, 5, 4, 5,5,5,4,4,5,5,4] #walk for 1-22 in 127
#invalid_rows = [0,0,4,5,4,3,5, 4] #empty for 1-7 in new location127
#invalid_rows = [0,4,3,4,5,5,4, 4] #sit for 1-7 in new location127
#invalid_rows = [0,4,5,5,3,4,5, 5] #stand for 1-7 in new loaction127
#invalid_rows = [0,5,5,5,5,5,3] #walk for 1-6 in new location127
#invalid_rows = [0,0,4,5,4,3,5, 4] #empty for 1-7 in new person127
#invalid_rows = [0,5,4,5,5,4,4] #sit for 1-7 in new person127
#invalid_rows = [0,5,2,5,5,3,3] #standfor 1-7 in new person127
invalid_rows = [0,5,5,4,4,5,4] #standfor 1-7 in new person127
base_path_input = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/origin/'
base_path_output = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/delete/'
for ex in range(1, 7):

    file_in = f'walk{ex}.csv'
    file_out = base_path_output + 'd' + file_in
    file_in = base_path_input + file_in


# Assumes first lines are correct. Missing Lines: Will copy from last valid CSI

    t = []
    sens = []
    dt = []
    tin = []
    tCSI1 = []
    tCSI2 = []
    tdma = []
    with open(file_in, "r", newline='') as f_input:
        reader = csv.reader(f_input)
        next(reader)
        lines = list(itertools.islice(reader, invalid_rows[ex], None))
        for row in lines:

            sens.append(row[1])
            t.append(float(row[13]))
            dt.append(float(row[14]))
            tin.append(float(row[15]))
            tCSI1.append(float(row[10]))
            tCSI2.append(float(row[11]))
            tdma.append(float(row[12]))

    seq = ['3', '4', '5', '6', '7']

    seq = seq[0:5]

    sq = np.array(seq)
    se = list(sens)
    incycle = []
    cIns = 0  # count Insertions
    cDel = 0

    rrr = NS * int(len(t) // NS)
    print(rrr, len(t))
    delList = list([])
    #    del lines[0:5]
    for j in range(NS, rrr, NS):  # remove "late" CSI lines
        for i in range(NS):
            tol1 = t[j] - tCSI1[i + j]
            tol2 = t[j] - tCSI2[i + j]
            tol3 = tdma[i + j] - 100 * i - 20
            if tol1 > 750 or tol2 > 750 or tol3 > 500:
                print('line', i + j, 'S', sens[i + j], '\ttCSI1:', tCSI1[i + j], '\ttCSI2:',
                      tCSI2[i + j], '\tt', t[j], '\ttol1:{:.2f}'.format(tol1), '\ttol2:{:.2f}'.format(tol2),
                      '\ttol3:{:.2f}'.format(tol3))
                if i + j < len(lines):
                    del lines[i + j]
                    delList = np.append(delList, int(i + j))
                    cDel += 1
        #       print('cIns:',cIns,i,se[i-1:i],seq[incycle])
    print('file_in:', file_in, 'Lines removed:', cDel)

    # delList = list(np.int(delList))
    # del lines[delList]

    with open(file_out, "w", newline='') as f_output:
        writer = csv.writer(f_output)
        writer.writerows(lines)

    f_input.close()
    f_output.close()

    print(' ')