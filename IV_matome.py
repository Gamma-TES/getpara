import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft
from scipy.optimize import curve_fit
import getpara as gp
import glob
from natsort import natsorted
import os
import sys
import re
import json
import matplotlib.cm as cm


R_SH = 3.9e-3  # shant resistance


# one degree func
def func(x, a, b):
    return a * x + b


def main():
    path = input("path: ")
    calib = input("calibration?[1] ")
    os.chdir(path)

    if calib == "1":
        files = natsorted(glob.glob("output/*.txt"))
    else:
        files = natsorted(glob.glob("rawdata/*.txt"))
    print(files)
    I_bias = []
    V_out = []

    cnt = 0
    for i in files:
        temp = re.sub(r"\D", "", i)
        data = np.loadtxt(i)
        I_bias = data[0]
        V_out = data[1]
        plt.plot(
            I_bias,
            V_out,
            marker="o",
            linewidth=1,
            markersize=4,
            label=f"{temp} mK",
            color=cm.hsv((float(cnt)) / float(len(files))),
        )
        cnt += 1
    plt.title(f"I-V ")
    plt.xlabel("I_bias[uA]")
    plt.ylabel("V_out[V]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/IV_calibration.png")
    plt.show()

    cnt = 0
    for i in files:
        temp = re.sub(r"\D", "", i)
        data = np.loadtxt(i)
        I_bias = data[0]
        V_out = data[1]

        popt, cov = curve_fit(func, I_bias[:10], V_out[:10])
        eta = 1 / popt[0]

        I_tes = eta * V_out
        I_sh = I_bias - I_tes
        V_tes = I_sh * R_SH
        R_tes = V_tes[1:] / I_tes[1:]
        R_tes = np.append(0.0, R_tes)

        plt.plot(
            I_bias,
            R_tes*1000,
            marker="o",
            linewidth=1,
            markersize=4,
            label=f"{temp} mK",
            color=cm.hsv((float(cnt)) / float(len(files))),
        )
        cnt += 1

    plt.title(f"I-R ")
    plt.xlabel("I_bias[uA]")
    plt.ylabel("R_tes[m$\Omega$]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/IR_calibration.png")
    plt.show()

    """
    print(I_bias[-10:-1])

    calib = '0'
    while calib == '0':
        # Fitting and get eta
        popt, cov = curve_fit(func,I_bias[:10],V_out[:10])
        eta = 1 / popt[0]
        print(f"eta (uA/V): {eta}")

        popt2, cov2 = curve_fit(func,I_bias[-10:-1],V_out[-10:-1])

        x_fit = np.arange(0,I_bias[-1],1)

        y_fit = func(x_fit,*popt2)
        y_fit -= y_fit[0]

        plt.plot(I_bias,V_out,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.plot(x_fit,func(x_fit,*popt))
        plt.plot(x_fit,y_fit)
        plt.title(f'I-V at {temp}')
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.show()
        
        start = int(input('start:'))
        stop = int(input('stop:'))

        mode = int(input("super:[0], normal[1], delete[2], else[3]:"))
        print(len(V_out))
        if mode == 0:
            diff =  func(start,*popt) - V_out[np.where(I_bias==start)][0]
            V_out[np.where(I_bias==start)[0][0]:np.where(I_bias==stop)[0][0] + 1 ] += diff
        elif mode == 1:
            
            diff =  y_fit[stop-1] - V_out[np.where(I_bias==stop)][0]
            V_out[np.where(I_bias==start)[0][0]:np.where(I_bias==stop)[0][0] + 1 ] += diff
        elif mode == 2:
            print(f"delete")
            V_out = np.delete(V_out,slice(np.where(I_bias==start)[0][0],np.where(I_bias==stop)[0][0] + 1 ))
            I_bias = np.delete(I_bias,slice(np.where(I_bias==start)[0][0],np.where(I_bias==stop)[0][0] + 1 )) 
        elif mode == 3:
            change = float(input('change:'))
            diff =  func(start,*popt) - V_out[np.where(I_bias==start)][0]
            V_out[np.where(I_bias==start)[0][0]:np.where(I_bias==stop)[0][0] + 1 ] += diff
        else:
            exit()
        
        plt.plot(I_bias,V_out,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.plot(x_fit,func(x_fit,*popt))
        plt.plot(x_fit,y_fit)
        plt.title(f'I-V at {temp}')
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.show()
        
        calib =  input('continue? yes[0],no[1]:')
    
    IV = [I_bias,V_out]
    np.savetxt(f"output/IV_{temp}_calibration.txt",IV)

    plt.plot(I_bias,V_out,marker = "o",c = "red",linewidth = 1,markersize = 6)
    plt.title(f'I-V at {temp}')
    plt.xlabel("I_bias[uA]")
    plt.ylabel("V_out[V]")
    plt.grid(True)
    plt.savefig(f"output/IV_{temp}_calibration.png")
    plt.show()
    """


if __name__ == "__main__":
    main()
