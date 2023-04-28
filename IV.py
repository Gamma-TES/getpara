
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
path = "/Volumes/Extreme 1TB/Matsumi/data/20230410/room2_ch2"
ch = 1 			# channel number
R_SH = 3.9e-3	# shant resistance

# set offset to zero
def offset(data):
    if data[0] > 0:
        data = data - data[0]
    else:
        data = data + data[0]
    
    if np.mean(data) < 0:
        data = data * -1
    return data

# one degree func
def func(x,a,b):
    return a * x + b
    

def main():
    
    os.chdir(path)
    temps = glob.glob("*mK")
    files = natsorted(glob.glob(os.path.join(temps[1],"*.dat")))
    
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
    

    I_tes = eta * V_out
    I_sh = I_bias - I_tes
    V_tes = I_sh * R_SH
    R_tes = V_tes[1:] / I_tes[1:]
    R_tes =  np.append(0.0,R_tes)

	# I-V graugh
    plt.plot(I_bias,V_out,marker = "o",c = "red",linewidth = 1,markersize = 6)
    plt.title('I-V at ')
    plt.xlabel("I_bias[uA]")
    plt.ylabel("V_out[V]")
    plt.grid(True)
    plt.show()
    
	# I-R graugh
    plt.plot(I_bias,R_tes,marker = "o",c = "red",linewidth = 1,markersize = 6)
    plt.title('I-R at ')
    plt.xlabel("I_bias[uA]",fontsize = 16)
    plt.ylabel("R_tes[$\Omega$]",fontsize = 16)
    plt.grid(True)
    plt.show()
    