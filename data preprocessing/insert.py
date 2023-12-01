################################################
# Insert lines to fullfill the standard sequence
################################################
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

N10s = 125  # 251 => 125 CSI Lines == 10 s

base_path_input = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/delete/d'
base_path_output = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/insert/'
for ex in range(1, 7):  # Looping from 5 to 30 for the filenames

    file_in = f'walk{ex}.csv'
    file_out = base_path_output + 'i' + file_in
    file_in = base_path_input + file_in

    t = []
    sens = []
    with open(file_in, "r", newline='') as f_input:
        reader = csv.reader(f_input)
        lines = list(reader)
        for row in lines:
            t.append(float(row[13]))
            sens.append(row[1])
            # lines.insert(5,lines[10])

        seq = ['3', '4', '5', '6', '7']

        seq = seq[0:NS]

        sq = np.array(seq)
        se = list(sens)
        incycle = []
        cIns = 0

        # print('t[0:10]',t[0:10])
        for i in range(len(t)):
            incycle = i % NS
            if se[i] != seq[incycle]:
                lines.insert(i, lines[i - NS])
                i = i - 1
                se.insert(i, seq[incycle])
                cIns += 1
        #       print('cIns:',cIns,i,se[i-1:i],seq[incycle])
        print('file_in:', file_in, 'Line insertions in the 1st run:', cIns)

        i = cIns
        staL = 0
        endL = len(t)
        nIns = cIns

        while cIns > 0 or nIns > 0:  # Proceed because cCins lines where inserted!
            staL = endL
            endL = endL + nIns
            cIns = nIns
            nIns = 0

            print('=======> First pass of len(t) processed')
            print(' ')
            for i in range(staL, endL):
                incycle = i % NS
                if se[i] != seq[incycle]:
                    lines.insert(i, lines[i - NS])
                    i = i - 1
                    se.insert(i, seq[incycle])
                    nIns += 1
                #                print('cIns:',cIns,'nIns:',nIns,i,se[i-1:i],seq[incycle])
                cIns -= 1
            print('file:', file_in, 'Line insertions in the next run, cIns:', cIns, 'new insertions nIns:', nIns)

        with open(file_out, "w", newline='') as f_output:
            writer = csv.writer(f_output)
            writer.writerows(lines)

        f_input.close()
        f_output.close()

        print(' ')