#write the funcion of aplitute over time
import matplotlib.pyplot as plt
import csv
import numpy as np
import cmath
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt

 ###########################
# LP Noise Reduction Filter
###########################   ampltitude over time
def amp_time (file_path,visualize):
    T = 1.0         # Sample Period
    fs = 12.5       # sample rate, Hz
    cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 3       # sin wave can be approx represented as quadratic
    n = int(T * fs) # total number of samples

    def butter_lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def check_and_fill_samples(y, target_samples=125, total_target_samples=1500):
        total_samples = len(y)
        if total_samples < total_target_samples:

            missing_samples = total_target_samples - total_samples
            last_valid_samples = y[-target_samples:]

            while missing_samples > 0:
                y.extend(last_valid_samples[:min(missing_samples, target_samples)])
                missing_samples -= target_samples
        elif total_samples > total_target_samples:
            # delete samples
            del y[total_target_samples:]

    N10s = 125
    NS = 5
    t = []
    y = []

    total_amp = np.array([])
    total_phase = np.array([])

    with open(file_path, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for row in lines:
            t.append(float(row[13]))
            y.append(row[16])
        check_and_fill_samples(y)
        total_frames = len(y) // N10s

        print("total_frames:", total_frames)
        print("Length of y:", len(y))

        for k in range(total_frames):
            CSi = []
            CSr = []

            # while t(n) < 10s *
                             #initial value，           end value                     step length
            for j in range(N10s * k, N10s * (k + 1), NS):  # One 10 seconds Frame
            #for j in range(N10s * k, min(N10s * (k + 1), len(y) - NS), NS):  # One 10 seconds Frame
                cvv = []   #store all csi data
                for i in range(NS):
                    cvv.append([float(item) for item in y[i + j][1:-1].split(',')])  #read array from each sensor, including 108 real and imaginary parts

                cv = np.array(cvv)      #change to numpy
                cv = cv.flatten()  # This converts the 2D array into a 1D array with length NS x 108*2. Since there are 5 sensors (NS=5), the length of cv is 1080.

            ####divide cv into real part（ci）and imaginary part（cr）：######
                ci = cv[0:len(cv):2]  # Frame Size = [N10s/Ns,(216/2 Im,Re || Amp,Pha)*NS]  length is(540,)
                cr = cv[1:len(cv):2]

                #######   remove pilot   #############
                idxRm = [ 26,53,80,107,134,161,188,215,242,269,296,323,350,377,404,431,458,485,512,539]

                for m in sorted(idxRm, reverse=True):
                    ci = np.delete(ci, m)
                    cr = np.delete(cr, m)

                CSi.append(ci)
                CSr.append(cr)

            ##################################

            imgI = np.array(CSi)    #(24,520)
            imgR = np.array(CSr)

            aa = np.array(imgI)
            ap = np.array(imgR)
            li, co = imgI.shape

            for i in range(li):
                for c in range(co):


                    aa[i][c] = abs(complex(imgI[i, c], imgR[i, c]))
                    ap[i][c] = cmath.phase(complex(imgI[i, c], imgR[i, c]))
                ######################################
                for i in range(li):
                    aa[i] = medfilt(aa[i])
                    ap[i] = medfilt(ap[i])

                #############    Phase sanitization
                # for i in range(li):
                #     for j in range(1, co):
                #         delta = ap[i, j] - ap[i, j - 1]  # if j > 0 else ap[i, j]
                #         if delta > np.pi:
                #             ap[i, j] = ap[i, j - 1] + delta - 2 * np.pi
                #         elif delta < -np.pi:
                #             ap[i, j] = ap[i, j - 1] + delta + 2 * np.pi

                for i in range(li):
                    aa[i] = butter_lowpass_filter(aa[i], cutoff, fs, 3)
                    ap[i] = butter_lowpass_filter(ap[i], cutoff, fs, 3)

            imgp = np.array(aa)  # all Amplitude in 10s
            realp = np.array(ap)

            if total_amp.size == 0:
                total_amp = imgp
                total_phase = realp
            else:
                total_amp = np.hstack((total_amp, imgp))
                total_phase = np.hstack((total_phase, realp))

        direction_amplitudes={}
        total_duration = total_frames * 10
        time_axis = np.linspace(0, total_duration, total_amp.shape[1])
        ###########################################################################################################

        seconds_per_direction = 10
        samples_per_second = 12.5
        samples_per_direction = int(seconds_per_direction * samples_per_second)
        directions = ["0 o'clock", "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock",
                      "5 o'clock", "6 o'clock", "7 o'clock", "8 o'clock", "9 o'clock",
                      "10 o'clock", "11 o'clock"]

        all_avg_amplitudes = []

        # Iterate over each direction
        for k, dir_name in enumerate(directions):
            # Extract data for the current direction
            start_idx = k * samples_per_direction
            end_idx = start_idx + samples_per_direction
            current_direction_amp = total_amp[:, start_idx:end_idx]
            current_direction_phase = total_phase[:, start_idx:end_idx]

            # Compute the average amplitude and phase for this direction
            avg_amp_for_direction = np.mean(current_direction_amp, axis=0)
            avg_phase_for_direction = np.mean(current_direction_phase, axis=0)

            # Store the data in the dictionary with direction name as the key
            direction_amplitudes[dir_name] = avg_amp_for_direction

            # Save the avg amplitude for plotting
            all_avg_amplitudes.append(avg_amp_for_direction)


        #Create a figure outside the direction loop
        if visualize:
            plt.figure(figsize=(10, 7))
            for k, dir_name in enumerate(directions):
                # Adjust the time axis for each direction
                time_offset = k * 10
                current_time_axis = np.linspace(time_offset, time_offset + 10, len(all_avg_amplitudes[k]))
                plt.plot(current_time_axis, all_avg_amplitudes[k], label=f"{dir_name}")

            plt.tick_params(axis='both', labelsize=22)
            plt.ylabel('|CSI|',fontsize=22)
            plt.xlabel('Time (s)',fontsize=22)
            plt.title(f"Amplitude over Time for All Directions", fontsize=22)
            plt.legend()
            plt.show()

        return {
            'time_axis': time_axis,
            'direction_amplitudes': direction_amplitudes
        }


##################test to show the image of amplitude over time for 12 directions###################



for ex in range(1, 2):
    file_in = f'walk{ex}.csv'
    file_in = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/0928itiv 326/insert/i' + file_in


    avg_amp = amp_time(file_in,visualize = True)
    print(avg_amp)
