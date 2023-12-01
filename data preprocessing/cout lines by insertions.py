# count lines before and after CSI line insertions
import matplotlib.pyplot as plt
import itertools
import csv
import os
import pandas
import numpy as np
import math
import cmath
NS= 5
for ex in range(1, 7):

    file_in = f'empty{ex}.csv'

    file_ins = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/insert/i' + file_in
    file_in = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/1019itiv 127/person/origin/'+ file_in


    t = []
    ti = []  #after inserion
    sens = []
    sensi = []  #after insertion

    with open(file_in, "r", newline='') as f_input:
        reader = csv.reader(f_input)
        next(reader)
        lines = list(reader)[5:]
        for row in lines:
            t.append(float(row[13]))
            sens.append(row[1])

    with open(file_ins, "r", newline='') as f_insput:
        reader = csv.reader(f_insput)
        lines = list(reader)
        for row in lines:
            ti.append(float(row[13]))
            sensi.append(row[1])

    # print(file_in, 'len(t):', len(t), file_ins, 'len(ti):', len(ti), '+:{:.3f}%'.format(100 * (len(ti) - len(t)) / len(t)))   #查看插值所占百分比
    print('+:{:.3f}%'.format(100 * (len(ti) - len(t)) / len(t)))

    f_input.close()
    f_insput.close()
