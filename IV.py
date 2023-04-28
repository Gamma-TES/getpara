
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack as fft
from scipy.optimize import curve_fit
import libs.getpara as gp
import glob
from natsort import natsorted
import os
import json

#-----------パラメタの設定----------------------
path = "E:/matsumi/data/20230428_3/room2-ch2"
ch = 1 			# channel number
R_SH = 3.9e-3	# shant resistance

# set offset to zero
def offset(data):
    if data[0] > 0:
        data = data - data[0]
    else:
        data = data + data[0]
    
    if np.mean(data[:10]) < 0:
        data = data * -1
    return data

# one degree func
def func(x,a,b):
    return a * x + b

def calibration(x,data,a,b):
    calib = '0'
    while calib == '0':
        x_fit = np.arange(0,800,1)

        plt.plot(x,data,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.plot(x_fit,func(x_fit,a,b))
        plt.title('I-V at ')
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.show()
        
        start = int(input('start:'))
        stop = int(input('stop:'))
        print(f"maybe {func(start,a,b)}")
        change = float(input('change: '))
        diff =  change - data[np.where(x==start)][0]

        data[np.where(x==start)[0][0]:np.where(x==stop)[0][0]] += diff

        plt.plot(x,data,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.plot(x_fit,func(x_fit,a,b))
        plt.title('I-V at ')
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.show()

        calib =  input('continue? yes[0],no[1]')

    return data



def main():
    
    os.chdir(path)
    os.makedirs('output',exist_ok = True)
    temps = glob.glob("*mK")

    for t in temps:
        print(t)
        files = natsorted(glob.glob(os.path.join(t,"*.dat")))
        I_bias = [] 
        V_out = []
        for i in files:
            data = gp.loadtxt(i)
            V_out.append(np.mean(data))
            I = os.path.splitext(os.path.basename(i))[0][10:-2]
            I_bias.append(int(I))
        
        V_out = np.array(V_out)
        I_bias = np.array(I_bias)
        

        V_out = offset(V_out)

        # Fitting and get eta
        popt, cov = curve_fit(func,I_bias[:10],V_out[:10])
        eta = 1 / popt[0]
        print(f"eta (uA/V): {eta}")

        popt2, cov2 = curve_fit(func,I_bias[-10:-1],V_out[-10:-1])
        

        I_tes = eta * V_out
        I_sh = I_bias - I_tes
        V_tes = I_sh * R_SH
        R_tes = V_tes[1:] / I_tes[1:]
        R_tes =  np.append(0.0,R_tes)

        x_fit = np.arange(0,I_bias[-1],1)
        y_fit = func(x_fit,*popt2) 
        y_fit -= y_fit[0]

        np.savetxt(f"output/IV_{t}.txt",[I_bias,V_out])

        # I-V graugh
        plt.plot(I_bias,V_out,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.plot(x_fit,y_fit)
        plt.title('I-V at {t}')
        plt.xlabel("I_bias[uA]")
        plt.ylabel("V_out[V]")
        plt.grid(True)
        plt.savefig(f"output/IV_{t}.png")
        plt.show()
        
        # I-R graugh
        plt.plot(I_bias,R_tes,marker = "o",c = "red",linewidth = 1,markersize = 6)
        plt.title('I-R at {t}')
        plt.xlabel("I_bias[uA]",fontsize = 16)
        plt.ylabel("R_tes[$\Omega$]",fontsize = 16)
        plt.grid(True)
        plt.savefig(f"output/IR_{t}.png")
        plt.cla()
        calibration(I_bias,V_out,*popt)

    for t in temps:
        files = natsorted(glob.glob(os.path.join(t,"*.dat")))
        I_bias = [] 
        V_out = []
        for i in files:
            data = gp.loadtxt(i)
            V_out.append(np.mean(data))
            I = os.path.splitext(os.path.basename(i))[0][10:-2]
            I_bias.append(int(I))
        
        V_out = np.array(V_out)
        I_bias = np.array(I_bias)
        V_out = offset(V_out)
        plt.plot(I_bias,V_out,marker = "o",linewidth = 1,markersize = 6,label=f'{t}mK')

    plt.title('I-V')
    plt.xlabel("I_bias[uA]")
    plt.ylabel("V_out[V]")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"output/IV_{t}mK.png")
    plt.show()
    
        

if __name__ == '__main__':
    main()